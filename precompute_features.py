#!/usr/bin/env python
"""Precompute MERT-based features and save them to disk to avoid repeated extraction during training.

Usage:
    python precompute_features.py --dataset deam --output_dir data/features/deam
    python precompute_features.py --dataset mtg-jamendo --output_dir data/features/mtg

This significantly speeds up training because MERT inference is the main bottleneck.
"""

import argparse
import os
import json
import csv
import hashlib
import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path


def load_mert_model(model_dir, device):
    """Load the MERT model."""
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return processor, model


def extract_features_for_audio(audio_path, processor, model, device, segment_duration=5, num_segments=3):
    """Extract all features for a single audio file."""
    # Load audio
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)
    
    # Resample to 24 kHz
    if sample_rate != 24000:
        resampler = T.Resample(sample_rate, 24000)
        waveform = resampler(waveform)
    
    segment_samples = segment_duration * 24000
    audio_length = waveform.shape[-1]
    
    # Extract num_segments random chunks using a fixed seed for consistency
    segments = []
    # Python's built-in hash() is randomized across processes/runs, so use a stable hash here.
    seed = int(hashlib.md5(audio_path.encode("utf-8")).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    for _ in range(num_segments):
        if audio_length >= segment_samples:
            start = np.random.randint(0, audio_length - segment_samples)
            segments.append(waveform[:, start:start + segment_samples])
        else:
            pad_length = segment_samples - audio_length
            padded = torch.nn.functional.pad(waveform, (0, pad_length))
            segments.append(padded)
    
    waveform = torch.cat(segments, dim=1)
    
    # Extract MERT features
    with torch.no_grad():
        inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=24000, return_tensors="pt", padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        mert = outputs.hidden_states[11].squeeze(0).cpu()  # Use layer 11 only
    
    # Extract mel features
    mel_transform = T.MelSpectrogram(
        sample_rate=24000,
        hop_length=320,
        n_fft=2048,
        n_mels=128,
        power=2.0
    )
    mel = mel_transform(waveform.squeeze(0))
    mel = torch.log(mel + 1e-9)
    mel = mel.transpose(1, 0)  # [Time, n_mels]
    
    # Extract chroma features
    waveform_np = waveform.squeeze(0).numpy()
    chroma = librosa.feature.chroma_cqt(y=waveform_np, sr=24000, hop_length=320)
    chroma = torch.tensor(chroma, dtype=torch.float32).transpose(1, 0)
    
    # Extract tempogram features
    tempogram = librosa.feature.tempogram(y=waveform_np, sr=24000, hop_length=320)
    tempogram = torch.tensor(tempogram, dtype=torch.float32).transpose(1, 0)
    
    return {
        'mert': mert.to(torch.float16),
        'mel': mel.to(torch.float16),
        'chroma': chroma.to(torch.float16),
        'tempogram': tempogram.to(torch.float16)
    }


def precompute_deam(output_dir, model_dir, device):
    """Precompute features for the DEAM dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_root = os.path.join(base_dir, "data", "DEAM", "audio")
    split_path = os.path.join(base_dir, "data", "DEAM", "deam_split.json")
    
    with open(split_path, "r") as f:
        split_data = json.load(f)
    
    processor, model = load_mert_model(model_dir, device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_ids = []
    for split in ['train', 'val', 'test']:
        all_ids.extend(split_data[split])
    all_ids = list(set(all_ids))
    
    print(f"Precomputing DEAM features ({len(all_ids)} audio files)...")
    
    for audio_id in tqdm(all_ids, desc="DEAM"):
        output_path = os.path.join(output_dir, f"{audio_id}.pt")
        
        # Skip files that already exist
        if os.path.exists(output_path):
            continue
        
        audio_path = os.path.join(audio_root, f"{audio_id}.mp3")
        if not os.path.exists(audio_path):
            print(f"Warning: audio file not found: {audio_path}")
            continue
        
        try:
            features = extract_features_for_audio(audio_path, processor, model, device)
            torch.save(features, output_path)
        except Exception as e:
            print(f"Error processing {audio_id}: {e}")
    
    print(f"DEAM features saved to {output_dir}")


def precompute_mtg(output_dir, model_dir, device):
    """Precompute features for MTG-Jamendo, saved separately by train/val/test split."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(base_dir, "data", "MTG-Jamendo", "mtg_labels.csv")
    audio_root = os.path.join(base_dir, "data", "MTG-Jamendo")
    
    processor, model = load_mert_model(model_dir, device)
    
    # Create train/val/test subdirectories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Collect all track records, including split information
    tracks = []
    with open(label_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip()
            if split not in {"train", "val", "test"}:
                continue

            track_id_raw = (row.get("track_id") or "").strip()
            try:
                track_id = int(track_id_raw)
            except ValueError:
                continue

            # mtg_labels.csv stores the real numeric audio path: {track_id%100}/{track_id}.low.mp3
            audio_path_rel = (row.get("audio_path") or "").strip()
            if not audio_path_rel:
                continue
            audio_path = os.path.join(audio_root, audio_path_rel)
            if os.path.exists(audio_path):
                tracks.append((split, track_id, audio_path))
    
    print(f"Precomputing MTG-Jamendo features ({len(tracks)} audio files)...")
    
    for split, track_id, audio_path in tqdm(tracks, desc="MTG-Jamendo"):
        # Save to the matching split subdirectory
        output_path = os.path.join(output_dir, split, f"{track_id}.pt")
        
        # Skip files that already exist
        if os.path.exists(output_path):
            continue
        
        try:
            features = extract_features_for_audio(audio_path, processor, model, device)
            torch.save(features, output_path)
        except Exception as e:
            print(f"Error processing {track_id}: {e}")
    
    print(f"MTG-Jamendo features saved to {output_dir} (organized by train/val/test)")


def main():
    parser = argparse.ArgumentParser(description="Precompute audio features")
    parser.add_argument('--dataset', type=str, required=True, choices=['deam', 'mtg-jamendo', 'all'],
                        help='Dataset to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data/features/<dataset>)')
    parser.add_argument('--model_dir', type=str, default='MERT',
                        help='MERT model directory')
    args = parser.parse_args()
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset == 'deam' or args.dataset == 'all':
        output_dir = args.output_dir or os.path.join(base_dir, "data", "features", "deam")
        precompute_deam(output_dir, args.model_dir, device)
    
    if args.dataset == 'mtg-jamendo' or args.dataset == 'all':
        output_dir = args.output_dir or os.path.join(base_dir, "data", "features", "mtg")
        precompute_mtg(output_dir, args.model_dir, device)


if __name__ == "__main__":
    main()
