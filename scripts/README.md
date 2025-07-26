# Privacy-Utility Evaluation Scripts

This directory contains the complete evaluation pipeline for assessing privacy protection and utility preservation after video anonymization.

## 📁 Organization

```
scripts/
├── README.md                           # This file
├── comprehensive_evaluation.py         # 🎯 MAIN SCRIPT - Unified evaluation
├── extract_frames.py                   # Video frame extraction
├── re-id/                              # 🔐 Privacy Assessment
│   ├── README.md                       # Privacy evaluation documentation
│   ├── evaluate_reid_osnet.py          # Main privacy evaluation script
│   ├── evaluate_identity_drift_osnet.py # Temporal consistency analysis
│   ├── evaluate_identity_drift_agw.py  # AGW-based drift analysis
│   ├── extract_person_crops_yolov5.py  # Person crop extraction
│   ├── organize_reid_dataset.py        # Dataset organization
│   └── visualize_identity_drift.py     # Drift visualization
└── utility/                            # 🎬 Utility Assessment
    ├── README.md                       # Utility evaluation documentation
    ├── evaluate_action_recognition_utility.py # Main utility evaluation script
    └── evaluation_vid_quality.py       # Video quality metrics
```

## 🎯 Quick Start

### Option 1: Comprehensive Evaluation (Recommended)
```bash
# Run complete privacy-utility evaluation
python comprehensive_evaluation.py --original_videos /path/to/original --anonymized_videos /path/to/anonymized
```

### Option 2: Individual Assessments
```bash
# Privacy Assessment
python re-id/evaluate_reid_osnet.py --query_dir /path/to/anonymized --gallery_dir /path/to/original

# Utility Assessment  
python utility/evaluate_action_recognition_utility.py --original_dir /path/to/original --anonymized_dir /path/to/anonymized
```

## 🔐 Privacy Assessment (Re-ID)

**Purpose**: Measure identity leakage after anonymization

**Key Metrics**:
- **Rank-1 Accuracy**: Percentage of anonymized queries correctly matched to original identities
- **mAP**: Mean Average Precision across all ranks

**Privacy Levels**:
- 🟢 EXCELLENT (< 10%)
- 🟡 GOOD (10-25%)
- 🟠 MODERATE (25-50%)
- 🔴 POOR (> 50%)

**Main Script**: `re-id/evaluate_reid_osnet.py`

## 🎬 Utility Assessment (Action Recognition)

**Purpose**: Measure utility preservation for downstream tasks

**Key Metrics**:
- **Top-1 Accuracy Retention**: Percentage of action recognition performance preserved

**Utility Levels**:
- 🟢 EXCELLENT (≥ 90%)
- 🟡 GOOD (75-90%)
- 🟠 MODERATE (50-75%)
- 🔴 POOR (< 50%)

**Main Script**: `utility/evaluate_action_recognition_utility.py`

## 📊 Comprehensive Evaluation

The `comprehensive_evaluation.py` script provides:

1. **Unified Pipeline**: Runs both privacy and utility assessment
2. **Combined Score**: Overall privacy-utility balance metric
3. **Clear Recommendations**: Actionable insights for improvement
4. **Detailed Reports**: JSON output with all results

**Output Levels**:
- 🟢 EXCELLENT (≥ 80% combined score)
- 🟡 GOOD (60-80% combined score)
- 🟠 MODERATE (40-60% combined score)
- 🔴 POOR (< 40% combined score)

## 🔧 Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV**: Video/image processing
- **NumPy**: Numerical computations

### Privacy Assessment
- **TorchReID**: Re-identification framework
- **OSNet**: Pre-trained person re-identification model
- **YOLOv5**: Person detection and cropping

### Utility Assessment
- **MMAction2**: Action recognition framework
- **TSN Model**: Pre-trained action recognition model

## 📈 Expected Results

### Good Anonymization Results
- **Privacy**: Rank-1 accuracy < 25% (GOOD or EXCELLENT)
- **Utility**: Accuracy retention ≥ 75% (GOOD or EXCELLENT)
- **Overall**: Combined score ≥ 60% (GOOD or EXCELLENT)

### Areas for Improvement
- **High Rank-1**: Strengthen anonymization method
- **Low Utility Retention**: Adjust anonymization parameters
- **Poor Combined Score**: Rebalance privacy-utility trade-off

## 🚀 Usage Examples

### Basic Evaluation
```bash
python comprehensive_evaluation.py \
  --original_videos datasets/original_videos \
  --anonymized_videos datasets/anonymized_videos
```

### With Ground Truth
```bash
python comprehensive_evaluation.py \
  --original_videos datasets/original_videos \
  --anonymized_videos datasets/anonymized_videos \
  --ground_truth datasets/ground_truth.csv
```

### Skip Preprocessing (if already done)
```bash
python comprehensive_evaluation.py \
  --original_videos datasets/original_videos \
  --anonymized_videos datasets/anonymized_videos \
  --skip_frames --skip_crops --skip_reid_org
```

## 📝 Output Files

- `comprehensive_evaluation_results.json`: Complete evaluation results
- `action_recognition_results.json`: Utility assessment results
- `temp_evaluation/`: Temporary processing files

## 🔗 Integration

This evaluation pipeline is designed to work with:
- **DeepPrivacy2**: Video anonymization system
- **MMAction2**: Action recognition framework
- **TorchReID**: Re-identification framework

For detailed documentation of individual components, see:
- `re-id/README.md`: Privacy assessment details
- `utility/README.md`: Utility assessment details 