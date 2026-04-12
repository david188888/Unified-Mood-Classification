# Unified Mood Classification

Unified multitask learning for music mood prediction:

- DEAM valence/arousal regression (continuous 2D targets)
- MTG-Jamendo multilabel mood tag classification

This README is written for the current runnable codebase (training, inference, and experiment scripts), not as a thesis narrative.

## What Is In This Repo

- Main training entry: `train.py`
- Feature precomputation entry: `precompute_features.py`
- Single-audio inference entry: `infer.py`
- Fast cached dataloader: `dataloader_fast.py`
- Model implementation: `src/models/`
- Experiment automation scripts: `scripts/train_full_suite.sh`, `scripts/train_feature_ablation.sh`

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `rich` is optional but recommended for better training progress UI.
- On macOS, training prefers MPS when available.

## Data Layout Expected By The Code

Required paths:

- `data/DEAM/audio/*.mp3`
- `data/DEAM/deam_split.json`
- `data/DEAM/annotations/annotations averaged per song/song_level/*.csv`
- `data/MTG-Jamendo/mtg_labels.csv`
- `data/MTG-Jamendo/<two-digit-folder>/*.low.mp3`

Cached feature output paths (generated):

- `data/features/deam/<id>.pt`
- `data/features/mtg/{train,val,test}/<track_id>.pt`

## Step 1: Precompute Features (Required)

Training in this repo uses cached features by default.

```bash
python precompute_features.py --dataset all
```

Options:

```bash
python precompute_features.py --dataset deam
python precompute_features.py --dataset mtg-jamendo
python precompute_features.py --dataset all --model_dir MERT
```

What is cached per sample:

- `mert` (MERT hidden layer 11)
- `mel`
- `chroma`
- `tempogram`

## Step 2: Train

### Baseline commands

```bash
# Early fusion baseline
python train.py --fusion_type early --batch_size 8 --epochs 50

# Late fusion baseline
python train.py --fusion_type late --batch_size 8 --epochs 50
```

### Resume from checkpoint

```bash
python train.py --resume runs/unified_mood_model_early/checkpoints/checkpoint_last.pt
```

If `--resume` is not provided, training auto-resumes from:

- `runs/unified_mood_model_{fusion_type}/checkpoints/checkpoint_last.pt`

Disable auto-resume:

```bash
python train.py --no_auto_resume
```

### Small subset run (debug/quick check)

```bash
python train.py --train_pct 0.1 --subset_seed 42 --epochs 2
```

### Useful options

- `--fusion_type {early,late}`
- `--enabled_features mert,mel,chroma,tempogram`
- `--batch_size`, `--epochs`, `--lr`
- `--num_workers`, `--eval_num_workers`
- `--accumulation_steps`
- `--compile` (PyTorch 2.x)
- `--plain_progress` or `--no_progress`

Feature ablation rule in current code:

- Early fusion: `mert` must be enabled.
- Late fusion: currently requires the full feature set.

## Step 3: Inference

Run inference on a single audio file:

```bash
python infer.py \
  --audio_path data/DEAM/audio/1769.mp3 \
  --checkpoint runs/experiment_suite_20260308_130926/early_baseline/unified_mood_model_early/checkpoints/checkpoint_best.pt \
  --task auto
```

Task modes:

- `--task auto` (resolve based on known dataset sample)
- `--task deam`
- `--task mtg`
- `--task both`

If cached features for the sample do not exist, inference can compute features on the fly using `--model_dir`.

## Model Structure (Current Implementation)

Defined in `src/models/unified_model.py` and related modules:

1. Feature projection (`feature_projection.py`)
2. Feature fusion (`feature_fusion.py`)
3. Conformer encoder(s) (`conv_transformer.py`)
4. Task heads (`output_heads.py`)

Fusion behavior:

- Early fusion: one fused stream then one encoder.
- Late fusion: one encoder per feature stream, then average encoded outputs.

Default model hyperparameters from `train.py`:

- `hidden_dim=512`
- `num_transformer_layers=4`
- `num_heads=8`

DEAM output ranges in model initialization:

- valence range: `(1.6, 8.4)`
- arousal range: `(1.6, 8.2)`

## Training Outputs

Per run directory:

- `runs/unified_mood_model_{fusion_type}/training.log`
- `runs/unified_mood_model_{fusion_type}/metrics.csv`
- `runs/unified_mood_model_{fusion_type}/checkpoints/checkpoint_last.pt`
- `runs/unified_mood_model_{fusion_type}/checkpoints/checkpoint_best.pt`
- TensorBoard event files under the same run directory

Visualize TensorBoard:

```bash
tensorboard --logdir runs/
```

## Common Experiment Workflows

### 1) Full experiment suite (early + late + ablations)

```bash
bash scripts/train_full_suite.sh
```

Useful environment variables:

- `EPOCHS` (default `30`)
- `BATCH_SIZE` (default `8`)
- `LATE_BATCH_SIZE` (default `4`)
- `NUM_WORKERS` (default `2`)
- `RESUME_EXISTING` (default `1`)
- `START_FROM` (resume from a named experiment step)

### 2) Early-fusion feature ablation only

```bash
bash scripts/train_feature_ablation.sh
```

### 3) Plot metrics from one run

```bash
python scripts/plot_metrics.py --metrics runs/unified_mood_model_early/metrics.csv
```

### 4) Generate thesis/experiment figures from saved runs

```bash
python scripts/generate_main_experiment_figures.py
python scripts/generate_feature_ablation_figures.py
python scripts/generate_thesis_figures.py
python scripts/generate_thesis_figures_v2.py
```

## Tests

```bash
python -m unittest discover -s test -v
```

Or run specific test files:

```bash
python -m unittest test.test_model_structure -v
python -m unittest test.test_data -v
python -m unittest test.test_infer -v
```

## Practical Notes / Pitfalls

- Precompute features before training, or the training pipeline will not have cached inputs.
- On macOS, high `num_workers` can cause file descriptor pressure; `--eval_num_workers 0` is often safer.
- Late fusion is more memory-heavy (multiple encoders); use a smaller batch size if needed.
- `requirements.txt` currently contains duplicate `nnAudio` entries; the last one installed by pip wins.
- Inference with `--task auto` only works when the input audio matches a known DEAM/MTG sample path in this repository.

## Minimal Quick Start

```bash
pip install -r requirements.txt
python precompute_features.py --dataset all
python train.py --fusion_type early --batch_size 8 --epochs 50
python infer.py --task auto
```
