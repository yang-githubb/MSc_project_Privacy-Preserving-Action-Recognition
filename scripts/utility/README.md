# Utility Assessment Pipeline

This folder contains scripts for evaluating **utility preservation** after video anonymization. The goal is to measure how well anonymized videos maintain their usefulness for downstream tasks like action recognition.

## 🎯 Purpose

Utility assessment measures whether anonymized videos can still be used effectively for their intended purpose. This is crucial for the privacy-utility trade-off analysis.

## 📁 Scripts

### Main Utility Assessment

#### `evaluate_action_recognition_utility.py` ⭐ **MAIN SCRIPT**
- **Purpose**: Evaluate action recognition performance preservation after anonymization
- **Input**: Original vs anonymized video clips
- **Model**: Pre-trained action recognition model (MMAction2 TSN)
- **Metrics**: Top-1 accuracy retention
- **Usage**:
  ```bash
  python evaluate_action_recognition_utility.py --original_dir /path/to/original --anonymized_dir /path/to/anonymized
  ```

#### `evaluation_vid_quality.py`
- **Purpose**: Assess video quality metrics after anonymization
- **Metrics**: SSIM, PSNR, LPIPS
- **Usage**:
  ```bash
  python evaluation_vid_quality.py --original_dir /path/to/original --anonymized_dir /path/to/anonymized
  ```

#### `evaluation_face_iden.py`
- **Purpose**: Evaluate face identification performance after anonymization
- **Metrics**: Face detection rate, identification accuracy
- **Usage**:
  ```bash
  python evaluation_face_iden.py --original_dir /path/to/original --anonymized_dir /path/to/anonymized
  ```

## 📊 Utility Metrics

### Action Recognition Utility
- **Top-1 Accuracy Retention**: Percentage of action recognition performance preserved
- **Utility Levels**:
  - 🟢 EXCELLENT (≥ 90%)
  - 🟡 GOOD (75-90%)
  - 🟠 MODERATE (50-75%)
  - 🔴 POOR (< 50%)

### Video Quality Metrics
- **SSIM**: Structural Similarity Index (higher = better)
- **PSNR**: Peak Signal-to-Noise Ratio (higher = better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower = better)

## 🎬 Evaluation Process

1. **Load Pre-trained Model**: MMAction2 TSN model trained on Kinetics-400
2. **Process Original Videos**: Extract action recognition predictions
3. **Process Anonymized Videos**: Extract action recognition predictions
4. **Compare Performance**: Calculate accuracy retention
5. **Generate Report**: Utility assessment with recommendations

## 🔧 Dependencies

- **MMAction2**: Action recognition framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Video processing
- **NumPy**: Numerical computations

## 📈 Interpretation

**High Utility Retention (≥ 75%)** ✅
- Anonymization preserves video utility well
- Actions remain recognizable after anonymization
- Good balance between privacy and utility

**Low Utility Retention (< 75%)** ⚠️
- Anonymization may be affecting video utility
- Actions are being obscured too much
- Consider adjusting anonymization parameters

## 🔗 Integration

This utility assessment is designed to work with the privacy assessment (Re-ID) to provide a complete privacy-utility trade-off analysis. Use the comprehensive evaluation script in the parent directory to run both assessments together. 