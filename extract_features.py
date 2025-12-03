#!/usr/bin/env python3
"""Feature extraction script for audio mood classification"""

import os
import sys
import h5py
import json
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
from torch_audiomentations import AddColoredNoise
from torchaudio.transforms import MelSpectrogram
import librosa

# Global configuration
CONFIG = {
    "sample_rate": 24000,
    "hop_length": 320,  # MERT cumulative stride, 13.33ms/frame
    "chunk_duration": 30,  # seconds
    "noise_probability": 0.2,
    "specaug_probability": 0.2,
    "mert_layers": [11, 12],  # 10th and 11th encoder layers
    "mel_params": {
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 320
    },
    "chroma_params": {
        "hop_length": 320
    },
    "tempogram_params": {
        "hop_length": 320
    }
}

_MODEL_CACHE = {
    "dir": None,
    "processor": None,
    "model": None
}

def load_mert_model(model_dir):
    """Load (or reuse) the MERT model and processor from local directory"""
    global _MODEL_CACHE

    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["dir"] == model_dir:
        return _MODEL_CACHE["processor"], _MODEL_CACHE["model"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

    if torch.backends.mps.is_available():
        model = model.to("mps")

    model.eval()
    _MODEL_CACHE = {
        "dir": model_dir,
        "processor": processor,
        "model": model
    }
    return processor, model

def resample_audio(audio, original_sr, target_sr=24000):
    """Resample audio to target sampling rate"""
    if original_sr != target_sr:
        resampler = T.Resample(original_sr, target_sr)
        audio = resampler(audio)
    return audio

def add_gaussian_noise(audio, sample_rate, probability=0.2):
    """Add Gaussian noise to audio with given probability"""
    if np.random.random() < probability:
        # White noise has f_decay 0.0
        augmenter = AddColoredNoise(min_snr_in_db=35, max_snr_in_db=38, p=1.0, min_f_decay=0.0, max_f_decay=0.0, output_type='tensor')
        # Ensure input is 3D: [batch_size, num_channels, num_samples]
        if audio.dim() == 2:  # [num_channels, num_samples]
            audio = audio.unsqueeze(0)  # [1, num_channels, num_samples]
        audio = augmenter(audio, sample_rate=sample_rate)
        # Remove batch dimension if it was added
        if audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)  # [num_channels, num_samples]
    return audio

def split_chunks(audio, sample_rate, chunk_duration=30):
    """Split audio into fixed-length chunks with silence padding"""
    chunk_samples = chunk_duration * sample_rate
    audio_length = audio.shape[-1]

    # Calculate number of chunks
    num_chunks = (audio_length + chunk_samples - 1) // chunk_samples

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio[:, start:end] if audio.dim() == 2 else audio[start:end]

        # Pad if shorter than chunk duration
        if chunk.shape[-1] < chunk_samples:
            pad_length = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_length))

        chunks.append(chunk)

    return torch.stack(chunks)

def extract_mert_features(model, processor, audio_chunks, sample_rate, batch_size=6):
    """Extract MERT features from audio chunks with batching to avoid OOM"""
    mert_features = []
    with torch.no_grad():
        for i in range(0, len(audio_chunks), batch_size):
            batch_chunks = audio_chunks[i:i+batch_size]
            inputs = processor(batch_chunks.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=False)

            # Move inputs to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # Get selected layers and concatenate
            selected_layers = [outputs.hidden_states[layer_idx] for layer_idx in CONFIG["mert_layers"]]
            batch_features = torch.cat(selected_layers, dim=-1)  # [B, T, D]
            # Move features back to CPU for further processing
            batch_features = batch_features.cpu()
            mert_features.append(batch_features)

    # Concatenate all batches
    return torch.cat(mert_features, dim=0) if mert_features else None

def process_single_chunk(chunk, mel_transform, sample_rate, config):
    """Process a single audio chunk to extract low-mid features"""
    # Add channel dimension if missing
    chunk = chunk.unsqueeze(0) if chunk.dim() == 1 else chunk

    # Mel
    mel = mel_transform(chunk)  # [1, n_mels, Time]
    mel = torch.log(mel + 1e-9)  # Log mel
    mel = mel.squeeze(0)  # [n_mels, Time]
    mel = apply_specaugment(mel, config["specaug_probability"])  # Apply SpecAugment before transposing
    mel = mel.transpose(1, 0)  # [Time, n_mels]

    # Chroma (using CQT as specified)
    chroma = librosa.feature.chroma_cqt(
        y=chunk.squeeze(0).numpy(),
        sr=sample_rate,
        hop_length=config["chroma_params"]["hop_length"]
    )
    chroma = torch.tensor(chroma).transpose(1, 0)  # [T, D] - already squeezed from librosa

    # Tempogram (using librosa)
    tempogram = librosa.feature.tempogram(
        y=chunk.squeeze(0).numpy(),
        sr=sample_rate,
        hop_length=config["tempogram_params"]["hop_length"]
    )
    tempogram = torch.tensor(tempogram, dtype=torch.float32).transpose(1, 0)  # force float32 to save memory

    return {
        "mel": mel,  # Already [T, D] after transpose
        "chroma": chroma,  # [T, D]
        "tempogram": tempogram  # [T, D]
    }

def extract_low_mid_features(audio_chunks, sample_rate):
    """Extract Mel-Spectrogram, Chroma, and Tempogram features"""
    # Mel-Spectrogram transform
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        **CONFIG["mel_params"]
    )

    features = []

    # Process chunks sequentially to reduce CPU usage
    for chunk in audio_chunks:
        features.append(process_single_chunk(chunk, mel_transform, sample_rate, CONFIG))

    return features

def apply_specaugment(mel_features, probability=0.2):
    """Apply SpecAugment to Mel-Spectrogram features"""
    if np.random.random() < probability:
        # Apply time masking
        time_mask = T.TimeMasking(time_mask_param=30)  # Max 30 frames mask
        mel_features = time_mask(mel_features)

        # Apply frequency masking
        freq_mask = T.FrequencyMasking(freq_mask_param=20)  # Max 20 Mel bin mask
        mel_features = freq_mask(mel_features)

    return mel_features

def store_audio_features(hf, audio_id, audio_features):
    """Store a single audio's features under its own group with float16 and compression"""
    group = hf.create_group(audio_id)

    mert_chunks = [cf["mert"] for cf in audio_features] if "mert" in audio_features[0] else []
    mel_chunks = [cf["mel"] for cf in audio_features]
    chroma_chunks = [cf["chroma"] for cf in audio_features]
    tempogram_chunks = [cf["tempogram"] for cf in audio_features]

    if mert_chunks:
        mert_tensor = torch.stack(mert_chunks)
        # Convert to float16 + gzip compression
        group.create_dataset("mert", data=mert_tensor.numpy().astype(np.float16), compression="gzip", compression_opts=6)

    mel_tensor = torch.stack(mel_chunks)
    group.create_dataset("mel", data=mel_tensor.numpy().astype(np.float16), compression="gzip", compression_opts=6)

    chroma_tensor = torch.stack(chroma_chunks)
    group.create_dataset("chroma", data=chroma_tensor.numpy().astype(np.float16), compression="lzf")

    tempogram_tensor = torch.stack(tempogram_chunks)
    group.create_dataset("tempogram", data=tempogram_tensor.numpy().astype(np.float16), compression="lzf")

def process_single_audio(audio_path, model_dir, config):
    """Process a single audio file and return its features"""
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]

    # Load model for this process
    processor, model = load_mert_model(model_dir)

    print(f"Processing: {audio_path}")

    # Load audio
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension

    # Resample
    waveform = resample_audio(waveform, sample_rate, config["sample_rate"])

    # Add noise
    waveform = add_gaussian_noise(waveform, config["sample_rate"], config["noise_probability"])

    # Split into chunks
    chunks = split_chunks(waveform, config["sample_rate"], config["chunk_duration"])

    # Extract features
    # MERT expects shape [B, T] for mono audio, squeeze the channel dimension
    mert_features = extract_mert_features(model, processor, chunks.squeeze(1), config["sample_rate"])
    low_mid_features = extract_low_mid_features(chunks, config["sample_rate"])

    # Combine features
    audio_features = []
    for i in range(len(chunks)):
        chunk_features = {
            "mert": mert_features[i],  # [T, D]
            "mel": low_mid_features[i]["mel"],  # SpecAugment already applied in extract_low_mid_features
            "chroma": low_mid_features[i]["chroma"],
            "tempogram": low_mid_features[i]["tempogram"]
        }

        audio_features.append(chunk_features)

    return audio_id, audio_features

def main():
    """Main function"""
    import argparse
    import concurrent.futures

    parser = argparse.ArgumentParser(description="Audio feature extraction script")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--model_dir", required=True, help="Directory containing MERT model files")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted features")
    parser.add_argument("--max_workers", type=int, default=max(1, os.cpu_count() - 4), help="Maximum number of workers (default: total CPU cores - 2)")
    args = parser.parse_args()

    if torch.backends.mps.is_available() and args.max_workers != 1:
        print("MPS backend detected; forcing max_workers=1 to avoid multiple GPU contexts.")
        args.max_workers = 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all audio files
    audio_files = [
        os.path.join(args.audio_dir, filename)
        for filename in os.listdir(args.audio_dir)
        if filename.endswith((".wav", ".flac", ".mp3"))
    ]

    print(f"Found {len(audio_files)} audio files to process")

    output_h5 = os.path.join(args.output_dir, "features.h5")
    with h5py.File(output_h5, 'w') as hf:
        if args.max_workers == 1:
            for audio_path in audio_files:
                audio_id, audio_features = process_single_audio(audio_path, args.model_dir, CONFIG)
                store_audio_features(hf, audio_id, audio_features)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_audio = {
                    executor.submit(process_single_audio, audio_path, args.model_dir, CONFIG): audio_path
                    for audio_path in audio_files
                }
                for future in concurrent.futures.as_completed(future_to_audio):
                    audio_id, audio_features = future.result()
                    store_audio_features(hf, audio_id, audio_features)

    print(f"Features stored in: {output_h5}")

    # Store configuration
    config_path = os.path.join(args.output_dir, "feature_config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Configuration stored in: {config_path}")

if __name__ == "__main__":
    main()
