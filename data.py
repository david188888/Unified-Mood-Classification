#!/usr/bin/env python
"""Data loader for DEAM and MTG-Jamendo datasets."""

import os
import json
import h5py
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import numpy as np
from torch_audiomentations import AddColoredNoise
import librosa
from torch.utils.data import Dataset, DataLoader


def load_deam_config():
    """Load DEAM feature extraction configuration."""
    config_path = "data/features/DEAM_features/feature_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


# Global variables
CONFIG = load_deam_config()

# MERT model cache
_MERT_MODEL_CACHE = {
    "dir": None,
    "processor": None,
    "model": None
}


def load_mert_model(model_dir):
    """Load (or reuse) the MERT model and processor."""
    global _MERT_MODEL_CACHE

    if _MERT_MODEL_CACHE["model"] is not None and _MERT_MODEL_CACHE["dir"] == model_dir:
        return _MERT_MODEL_CACHE["processor"], _MERT_MODEL_CACHE["model"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

    if torch.backends.mps.is_available():
        model = model.to("mps")

    model.eval()
    _MERT_MODEL_CACHE = {
        "dir": model_dir,
        "processor": processor,
        "model": model
    }
    return processor, model


class DEAMDataset(Dataset):
    """DEAM Dataset with pre-extracted features."""

    def __init__(self, split="train", feature_path=None, label_path=None):
        self.split = split
        self.feature_path = feature_path or "/Users/david/codespace/Unified-Mood-Classification-Mamba/data/features/DEAM_features/features.h5"
        self.label_path = label_path or "/Users/david/codespace/Unified-Mood-Classification-Mamba/data/DEAM/annotations/annotations averaged per song/song_level/"

        # Load split IDs
        with open("/Users/david/codespace/Unified-Mood-Classification-Mamba/data/DEAM/deam_split.json", "r") as f:
            split_data = json.load(f)
            self.audio_ids = split_data[self.split]

        # Load labels
        self.labels = self._load_labels()

    def _load_labels(self):
        """Load and combine DEAM labels from two CSV files."""
        labels = {}

        # Load first CSV (1-2000)
        csv_path1 = os.path.join(self.label_path, "static_annotations_averaged_songs_1_2000.csv")
        with open(csv_path1, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        # Load second CSV (2000-2058)
        csv_path2 = os.path.join(self.label_path, "static_annotations_averaged_songs_2000_2058.csv")
        with open(csv_path2, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                song_id = parts[0]
                valence = float(parts[1])
                arousal = float(parts[3])
                labels[song_id] = (valence, arousal)

        return labels

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]

        # Read features from h5 file
        with h5py.File(self.feature_path, "r") as hf:
            group = hf[audio_id]
            mert = torch.tensor(group["mert"][()].astype(np.float16)).reshape(-1, group["mert"].shape[-1])  # [C*T, D]
            mel = torch.tensor(group["mel"][()].astype(np.float16)).reshape(-1, group["mel"].shape[-1])  # [C*T, D]
            chroma = torch.tensor(group["chroma"][()].astype(np.float16)).reshape(-1, group["chroma"].shape[-1])  # [C*T, D]
            tempogram = torch.tensor(group["tempogram"][()].astype(np.float16)).reshape(-1, group["tempogram"].shape[-1])  # [C*T, D]

        # Get label
        valence, arousal = self.labels[audio_id]
        label = torch.tensor([valence, arousal], dtype=torch.float32)

        return {
            "mert": mert,
            "mel": mel,
            "chroma": chroma,
            "tempogram": tempogram
        }, label


class MTGJamendoDataset(Dataset):
    """MTG-Jamendo Dataset with on-the-fly feature extraction."""

    def __init__(self, split="train", label_path=None, audio_root=None, model_dir=None):
        self.split = split
        self.label_path = label_path or "/Users/david/codespace/Unified-Mood-Classification-Mamba/data/MTG-Jamendo/mtg_split_labels.csv"
        self.audio_root = audio_root or "/Users/david/codespace/Unified-Mood-Classification-Mamba/data/MTG-Jamendo"
        self.model_dir = model_dir
        self.mood_tags = self._load_tags()
        self.data = self._load_data()

        # Initialize Mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=CONFIG["sample_rate"],
            **CONFIG["mel_params"]
        )

    def _load_tags(self):
        """Extract all unique mood tags."""
        tags = set()
        with open(self.label_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                mood_tag_list = parts[5].split("|")
                for tag in mood_tag_list:
                    tags.add(tag)
        return sorted(list(tags))

    def _load_data(self):
        """Load MTG-Jamendo data for the given split."""
        data = []
        with open(self.label_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                split, track_id, _, _, duration, mood_tags_str, _ = parts
                track_id = int(track_id)
                if split != self.split:
                    continue

                # Build audio path: data/MTG-Jamendo/{track_id%100}/{track_id}.low.mp3
                audio_subdir = f"{track_id % 100:02d}"
                audio_filename = f"{track_id}.low.mp3"
                audio_path = os.path.join(self.audio_root, audio_subdir, audio_filename)

                # Skip if audio file doesn't exist
                if not os.path.exists(audio_path):
                    continue

                mood_tags = mood_tags_str.split("|")

                data.append({
                    "track_id": track_id,
                    "audio_path": audio_path,
                    "mood_tags": mood_tags
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]

        # Load audio
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension

        # Resample
        if sample_rate != CONFIG["sample_rate"]:
            resampler = T.Resample(sample_rate, CONFIG["sample_rate"])
            waveform = resampler(waveform)

        # Add noise
        if np.random.random() < CONFIG["noise_probability"]:
            augmenter = AddColoredNoise(
                min_snr_in_db=35, max_snr_in_db=38, p=1.0,
                min_f_decay=0.0, max_f_decay=0.0, output_type='tensor'
            )
            # Ensure input is 3D: [batch_size, num_channels, num_samples]
            if waveform.dim() == 2:  # [num_channels, num_samples]
                waveform = waveform.unsqueeze(0)  # [1, num_channels, num_samples]
            waveform = augmenter(waveform, sample_rate=CONFIG["sample_rate"])
            # Remove batch dimension if it was added
            if waveform.dim() == 3 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)  # [num_channels, num_samples]

        # Convert to 45s audio
        target_samples = CONFIG["chunk_duration"] * CONFIG["sample_rate"]  # 45s * sample_rate
        audio_length = waveform.shape[-1]

        if audio_length > target_samples:
            # If longer than 45s, randomly crop a 45s segment
            random_start = torch.randint(0, audio_length - target_samples, (1,)).item()
            waveform = waveform[:, random_start:random_start + target_samples]
        elif audio_length < target_samples:
            # If shorter than 45s, pad to 45s
            pad_length = target_samples - audio_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Reshape to [1, 1, T] to maintain chunk format compatibility
        chunks = waveform.unsqueeze(0)  # [1, 1, T] where T is 45s worth of samples

        # Extract MERT features
        processor, model = load_mert_model(self.model_dir)
        mert_features = []
        with torch.no_grad():
            for chunk in chunks:
                # Squeeze channel dim: [1, T]
                chunk = chunk.squeeze(1)
                inputs = processor(chunk.numpy(), sampling_rate=CONFIG["sample_rate"], return_tensors="pt", padding=False)

                # Move inputs to model device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True)
                selected_layers = [outputs.hidden_states[layer_idx] for layer_idx in CONFIG["mert_layers"]]
                chunk_mert = torch.cat(selected_layers, dim=-1).squeeze(0)  # [T, D]
                mert_features.append(chunk_mert.cpu())
        mert = torch.cat(mert_features, dim=0)  # [T_total, D]

        # Extract mel features
        mel_chunks = []
        for chunk in chunks:
            # [1, T] -> [T]
            chunk = chunk.squeeze(0)
            mel = self.mel_transform(chunk)  # [n_mels, Time]
            mel = torch.log(mel + 1e-9)  # Log mel

            # Apply SpecAugment
            if np.random.random() < CONFIG["specaug_probability"]:
                time_mask = T.TimeMasking(time_mask_param=30)
                mel = time_mask(mel)

                freq_mask = T.FrequencyMasking(freq_mask_param=20)
                mel = freq_mask(mel)

            mel = mel.transpose(1, 0)  # [Time, n_mels]
            mel_chunks.append(mel)
        mel = torch.cat(mel_chunks, dim=0)  # [T_total, D]

        # Extract chroma
        chroma_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            chroma = librosa.feature.chroma_cqt(
                y=chunk_np,
                sr=CONFIG["sample_rate"],
                hop_length=CONFIG["chroma_params"]["hop_length"]
            )
            chroma = torch.tensor(chroma).transpose(1, 0)  # [T, D]
            chroma_chunks.append(chroma)
        chroma = torch.cat(chroma_chunks, dim=0)  # [T_total, D]

        # Extract tempogram
        tempogram_chunks = []
        for chunk in chunks:
            chunk_np = chunk.squeeze(0).numpy()
            tempogram = librosa.feature.tempogram(
                y=chunk_np,
                sr=CONFIG["sample_rate"],
                hop_length=CONFIG["tempogram_params"]["hop_length"]
            )
            tempogram = torch.tensor(tempogram, dtype=torch.float32).transpose(1, 0)  # [T, D]
            tempogram_chunks.append(tempogram)
        tempogram = torch.cat(tempogram_chunks, dim=0)  # [T_total, D]

        # Encode labels
        mood_tags = item["mood_tags"]
        label = torch.zeros(len(self.mood_tags), dtype=torch.float32)
        for tag in mood_tags:
            if tag in self.mood_tags:
                label[self.mood_tags.index(tag)] = 1.0

        return {
            "mert": mert,
            "mel": mel,
            "chroma": chroma,
            "tempogram": tempogram
        }, label


def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    # All features have the same shape across the batch in DEAM
    # For MTG-Jamendo, chunks are fixed to 30s, so total length varies based on number of chunks

    features_batch = {}
    labels = []

    # Separate features and labels
    all_features, all_labels = zip(*batch)

    # Stack each feature type
    for key in all_features[0].keys():
        features = [feat[key] for feat in all_features]
        # Pad sequences to the maximum length in the batch
        max_len = max(feat.shape[0] for feat in features)

        padded_features = []
        for feat in features:
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                padded = torch.nn.functional.pad(feat, (0, 0, 0, pad_len))  # Pad along time dimension
            else:
                padded = feat
            padded_features.append(padded)

        features_batch[key] = torch.stack(padded_features)

    # Stack labels
    labels = torch.stack(all_labels)

    return features_batch, labels


def get_dataloader(dataset_name, split="train", batch_size=8, shuffle=True, num_workers=0, **kwargs):
    """Get dataloader for the specified dataset."""
    if dataset_name == "deam":
        dataset = DEAMDataset(split=split, **kwargs)
    elif dataset_name == "mtg-jamendo":
        if "model_dir" not in kwargs:
            raise ValueError("model_dir is required for mtg-jamendo dataset")
        dataset = MTGJamendoDataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # Test DEAM dataloader
    print("Testing DEAM Dataloader...")
    deam_loader = get_dataloader("deam", split="train", batch_size=4)
    deam_batch = next(iter(deam_loader))
    deam_features, deam_labels = deam_batch
    print(f"DEAM Batch Features: {deam_features.keys()}")
    for key, val in deam_features.items():
        print(f"  {key}: {val.shape}")
    print(f"DEAM Batch Labels: {deam_labels.shape}")

    # Test MTG-Jamendo dataloader
    print("\nTesting MTG-Jamendo Dataloader...")
    try:
        mtg_loader = get_dataloader("mtg-jamendo", split="train", batch_size=2, model_dir="MERT", num_workers=0)
        mtg_batch = next(iter(mtg_loader))
        mtg_features, mtg_labels = mtg_batch
        print(f"MTG-Jamendo Batch Features: {mtg_features.keys()}")
        for key, val in mtg_features.items():
            print(f"  {key}: {val.shape}")
        print(f"MTG-Jamendo Batch Labels: {mtg_labels.shape}")
        print("MTG-Jamendo Dataloader test passed!")
    except Exception as e:
        print(f"MTG-Jamendo Dataloader test failed: {str(e)}")
