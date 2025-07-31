# Scripts Directory

This directory contains all evaluation and processing scripts for the privacy-utility assessment project.

## 📁 Directory Structure

```
scripts/
├── README.md                    # This file
├── evaluation/                  # Re-ID evaluation scripts
│   ├── README.md               # Evaluation documentation
│   ├── cpu_preprocessing.slurm # CPU preprocessing job
│   └── gpu_evaluation.slurm    # GPU evaluation job
├── re-id/                      # Re-ID processing scripts
│   ├── README.md               # Re-ID documentation
│   ├── videos_to_frames.py     # Frame extraction (with skip)
│   ├── crop_fullbody.py        # Person cropping (with skip)
│   ├── organize_reid_data.py   # Data organization
│   └── evaluate_reid_osnet.py  # Re-ID evaluation
├── dataset_prep/               # Dataset preparation utilities
├── utility/                    # Other utility scripts
```

## 🚀 Quick Start

### Re-ID Evaluation (Privacy Assessment)

```bash
# Step 1: CPU preprocessing (8 hours, no GPU)
sbatch scripts/evaluation/cpu_preprocessing.slurm

# Step 2: GPU evaluation (2 hours, efficient GPU usage)
sbatch scripts/evaluation/gpu_evaluation.slurm
```

## 📊 What Each Directory Does

### `evaluation/`
- **Purpose**: Organized Re-ID evaluation pipeline
- **Features**: CPU/GPU split, skip functionality, efficient resource usage
- **Scripts**: 
  - `cpu_preprocessing.slurm` - Frame extraction, cropping, organization
  - `gpu_evaluation.slurm` - Re-ID model evaluation only

### `re-id/`
- **Purpose**: Core Re-ID processing scripts
- **Features**: Skip functionality, parallel processing, optimization
- **Scripts**:
  - `videos_to_frames.py` - Extract frames from videos
  - `crop_fullbody.py` - Crop person images using pose detection
  - `organize_reid_data.py` - Organize data for Re-ID evaluation
  - `evaluate_reid_osnet.py` - Run OSNet Re-ID evaluation

### `dataset_prep/`
- **Purpose**: Dataset preparation and preprocessing utilities
- **Use**: When you need to prepare or modify datasets

### `utility/`
- **Purpose**: General utility scripts and tools
- **Use**: For various project utilities and helper functions

## ⚡ Key Features

1. **Efficient Resource Usage**: CPU preprocessing + GPU evaluation only
2. **Skip Functionality**: Won't redo completed work
3. **Time Savings**: 6 hours → 2 hours GPU time
4. **Clean Organization**: Each directory has specific purpose

## 📋 Output Structure

```
output/reid_evaluation/
├── original_frames/             # Extracted original video frames
├── anonymized_frames/           # Extracted anonymized video frames
├── original_crops/              # Cropped person images (original)
├── anonymized_crops/            # Cropped person images (anonymized)
├── reid_original/               # Organized Re-ID data (original)
├── reid_anonymized/             # Organized Re-ID data (anonymized)
├── reid_results.json            # Final evaluation results
└── preprocessing_complete.flag   # Preprocessing completion flag
```

## 🔍 Monitoring

- **Job Status**: `squeue -u $USER`
- **Logs**: `tail -f logs/eval/reid_*.out`
- **Progress**: Check individual directory READMEs for specific details

## 📖 Documentation

- **Main Evaluation**: See `evaluation/README.md`
- **Re-ID Processing**: See `re-id/README.md`
- **Dataset Prep**: See `dataset_prep/README.md` (if exists)
- **Utilities**: See `utility/README.md` (if exists) 