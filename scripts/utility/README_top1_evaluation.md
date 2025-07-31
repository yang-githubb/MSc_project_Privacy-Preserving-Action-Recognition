# Top-1 Accuracy Evaluation for DeepPrivacy2 Anonymization Impact

This directory contains scripts to evaluate how DeepPrivacy2 full-body anonymization affects task-level performance using Top-1 accuracy on a UCF101 subset.

## 📋 Overview

The evaluation uses MMAction2's TSN (Temporal Segment Networks) model to compare action recognition performance between original and anonymized videos. The goal is to measure the utility preservation of the anonymization process.

## 🎯 Evaluation Setup

### Requirements

1. **MMAction2 Installation**:
   ```bash
   pip install -U openmim
   mim install mmaction2
   ```

2. **UCF101 Dataset**: Original videos organized by class
3. **Anonymized Videos**: DeepPrivacy2 processed videos with same structure
4. **TSN Checkpoint**: Pre-trained model for UCF101 action recognition

## 📁 Scripts

### 1. `evaluate_top1_accuracy_tsn.py` - Main Evaluation Script

Evaluates Top-1 accuracy on original vs anonymized videos using TSN model.

**Usage**:
```bash
python scripts/utility/evaluate_top1_accuracy_tsn.py \
    --original_dir /path/to/original/ucf101/videos \
    --anonymized_dir /path/to/anonymized/ucf101/videos \
    --num_classes 10 \
    --videos_per_class 10 \
    --output_file results_top1_accuracy.json
```

**Parameters**:
- `--original_dir`: Path to original UCF101 videos
- `--anonymized_dir`: Path to anonymized UCF101 videos  
- `--ground_truth`: Optional CSV file with video_name,true_label format
- `--num_classes`: Number of classes to evaluate (default: 10)
- `--videos_per_class`: Videos per class to evaluate (default: 10)
- `--device`: Device for inference (default: cuda)
- `--output_file`: JSON file for detailed results

### 2. `download_tsn_checkpoint.py` - Checkpoint Downloader

Downloads the pre-trained TSN checkpoint for UCF101.

**Usage**:
```bash
python scripts/utility/download_tsn_checkpoint.py
```

### 3. `generate_ground_truth_csv.py` - Ground Truth Generator

Creates a CSV file with ground truth labels for the evaluation subset.

**Usage**:
```bash
python scripts/utility/generate_ground_truth_csv.py \
    --original_dir /path/to/original/ucf101/videos \
    --output_file ground_truth_ucf101_subset.csv \
    --num_classes 10 \
    --videos_per_class 10
```

## 🔧 Setup Instructions

### Step 1: Download TSN Checkpoint
```bash
cd /work/tc067/tc067/s2737744
python scripts/utility/download_tsn_checkpoint.py
```

### Step 2: Generate Ground Truth (Optional)
```bash
python scripts/utility/generate_ground_truth_csv.py \
    --original_dir Dataset/ucf101/UCF-101 \
    --output_file ground_truth_ucf101_subset.csv
```

### Step 3: Run Evaluation
```bash
python scripts/utility/evaluate_top1_accuracy_tsn.py \
    --original_dir Dataset/ucf101/UCF-101 \
    --anonymized_dir output/ucf101_anonymized \
    --ground_truth ground_truth_ucf101_subset.csv \
    --output_file results_top1_accuracy.json
```

## 📊 Expected Output

The evaluation will produce:

1. **Console Output**:
   ```
   ==================================================
   EVALUATING ORIGINAL VIDEOS
   ==================================================
   Original Top-1 Accuracy: 85.20% (85/100)

   ==================================================
   EVALUATING ANONYMIZED VIDEOS  
   ==================================================
   Anonymized Top-1 Accuracy: 78.00% (78/100)

   ==================================================
   EVALUATION SUMMARY
   ==================================================
   Total Videos Evaluated: 100
   Original Accuracy: 85.20%
   Anonymized Accuracy: 78.00%
   Absolute Accuracy Drop: 7.20%
   Relative Accuracy Drop: 8.45%
   Anonymization Impact: MODERATE
   ```

2. **JSON Results** (if `--output_file` specified):
   - Detailed per-video results
   - Per-class accuracy breakdown
   - Configuration and summary metrics

## 📈 Interpretation

The evaluation provides several metrics:

- **Top-1 Accuracy**: Percentage of correctly classified videos
- **Accuracy Drop**: Absolute and relative performance decrease
- **Impact Level**: 
  - MINIMAL: < 5% drop
  - MODERATE: 5-15% drop  
  - SIGNIFICANT: 15-30% drop
  - SEVERE: > 30% drop

## 🎯 Evaluation Subset

By default, the evaluation uses:
- **10 classes** from UCF101
- **10 videos per class** = 100 total videos
- **Balanced sampling** across action categories

This provides a representative sample while keeping evaluation time reasonable.

## 🔍 Per-Class Analysis

The script provides detailed per-class breakdown showing which action categories are most/least affected by anonymization:

```
Per-Class Analysis:
  ApplyEyeMakeup: 90.0% → 80.0% (drop: 10.0%)
  ApplyLipstick: 85.0% → 75.0% (drop: 10.0%)
  Basketball: 95.0% → 90.0% (drop: 5.0%)
  ...
```

## 🛠️ Troubleshooting

### Common Issues:

1. **MMAction2 Import Error**:
   ```bash
   pip install -U openmim
   mim install mmaction2
   ```

2. **Checkpoint Not Found**:
   ```bash
   python scripts/utility/download_tsn_checkpoint.py
   ```

3. **CUDA Out of Memory**:
   - Use `--device cpu` for CPU inference
   - Reduce batch size in config if needed

4. **No Video Pairs Found**:
   - Check directory structure matches expected format
   - Verify anonymized videos exist with `_anonymized.mp4` suffix

## 📝 Directory Structure

Expected structure:
```
original_dir/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   └── ...
├── ApplyLipstick/
│   └── ...

anonymized_dir/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01_anonymized.mp4
│   └── ...
├── ApplyLipstick/
│   └── ...
```

## 🎯 Use Cases

This evaluation is useful for:

1. **Privacy-Utility Trade-off Analysis**: Quantify how much task performance is preserved
2. **Anonymization Method Comparison**: Compare different anonymization approaches
3. **Dataset Preparation**: Assess if anonymized data is suitable for training
4. **Research Validation**: Validate anonymization effectiveness for action recognition

## 📚 References

- [MMAction2 Documentation](https://mmaction2.readthedocs.io/)
- [TSN Paper](https://arxiv.org/abs/1608.00859)
- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [DeepPrivacy2](https://github.com/hukkelas/DeepPrivacy2) 