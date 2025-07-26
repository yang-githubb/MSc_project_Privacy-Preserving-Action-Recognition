# Re-ID Evaluation Pipeline

This pipeline evaluates the effectiveness of anonymization by testing if anonymized images can still be matched to original identities using re-identification models.

## Overview

The pipeline consists of 4 main steps:
1. **Extract frames** from original and anonymized videos
2. **Crop and remove background** from frames to isolate persons
3. **Organize data** for re-ID evaluation
4. **Evaluate re-ID performance** using OSNet model

## Complete Pipeline Commands

### Step 1: Extract Frames from Videos
```bash
python scripts/re-id/videos_to_frames.py \
    --original_video /work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101/Archery/v_Archery_g01_c01.avi \
    --anonymized_video /work/tc067/tc067/s2737744/output/ucf101_anonymized/Archery/v_Archery_g01_c01_anonymized.mp4 \
    --output_dir datasets/video_frames
```

### Step 2: Crop Full Body and Remove Background
```bash
# Process original frames
python scripts/re-id/crop_fullbody.py \
    --input_dir datasets/video_frames/Archery_g01_c01/original_frames \
    --output_dir datasets/cropped/original_cropped \
    --remove_background

# Process anonymized frames
python scripts/re-id/crop_fullbody.py \
    --input_dir datasets/video_frames/Archery_g01_c01/anonymized_frames \
    --output_dir datasets/cropped/anonymized_cropped \
    --remove_background
```

### Step 3: Organize Data for Re-ID Evaluation
```bash
python scripts/re-id/organize_reid_data.py \
    --original_cropped_dir datasets/cropped/original_cropped \
    --anonymized_cropped_dir datasets/cropped/anonymized_cropped \
    --output_dir datasets/reid_eval
```

### Step 4: Run Re-ID Evaluation
```bash
python scripts/re-id/evaluate_reid_osnet.py \
    --query_dir datasets/reid_eval/query \
    --gallery_dir datasets/reid_eval/gallery
```

## Data Organization Strategy

The pipeline organizes data as follows:

- **Query**: 1 anonymized frame (ID 0001, camera c1)
- **Gallery**: 
  - Remaining anonymized frames (ID 0001, camera c1)
  - 1 original frame (ID 0001, camera c2)

This setup tests if the anonymized query can be matched to the original frame.

## Evaluation Metrics

The evaluation provides:
- **Rank-1 Accuracy**: Percentage of queries correctly matched at rank 1
- **mAP Score**: Mean Average Precision
- **Privacy Level**: 
  - EXCELLENT: < 10% Rank-1 accuracy
  - GOOD: 10-25% Rank-1 accuracy  
  - MODERATE: 25-50% Rank-1 accuracy
  - POOR: > 50% Rank-1 accuracy

## Interpretation

- **High Rank-1 accuracy** = Bad anonymization (identity still recognizable)
- **Low Rank-1 accuracy** = Good anonymization (identity properly obfuscated)

## Background Removal

The crop script includes background removal with white background:
- Uses MediaPipe Selfie Segmentation (preferred)
- Falls back to OpenCV DNN segmentation
- Replaces background with white (255, 255, 255)
- Enabled by default with `--remove_background`
- Can be disabled with `--no_remove_background`

## Batch Processing

For processing multiple videos, use batch mode:

```bash
# Batch frame extraction (future feature)
python scripts/re-id/videos_to_frames.py --batch_file video_pairs.txt

# Batch cropping
python scripts/re-id/crop_fullbody.py --batch_mode \
    --input_base_dir datasets/video_frames \
    --output_base_dir datasets/cropped

# Batch re-ID organization
python scripts/re-id/organize_reid_data.py --batch_mode \
    --cropped_base_dir datasets/cropped \
    --output_base_dir datasets/reid_eval_batch
```

## Dependencies

- OpenCV
- MediaPipe (for pose detection and background removal)
- PyTorch
- torchreid (for re-ID evaluation)
- PIL/Pillow
- NumPy

## File Structure

```
datasets/
├── video_frames/
│   └── Archery_g01_c01/
│       ├── original_frames/
│       └── anonymized_frames/
├── cropped/
│   ├── original_cropped/
│   └── anonymized_cropped/
└── reid_eval/
    ├── query/
    └── gallery/
```

## Notes

- Uses random seed 42 for reproducible frame selection
- Frame selection is randomized to avoid bias
- Background removal improves re-ID evaluation by removing distracting elements
- The pipeline is designed to test anonymization effectiveness specifically 