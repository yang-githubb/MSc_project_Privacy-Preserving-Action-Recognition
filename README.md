# Privacy-Preserving Video Anonymization with DeepPrivacy2

**MSc Project: Pose-Conditioned GANs with Temporal Consistency for Privacy-Preserving Video Anonymization**

## 🎯 Project Overview

This project implements a comprehensive privacy-preserving video anonymization system using DeepPrivacy2, a state-of-the-art pose-conditioned generative adversarial network (GAN). The system generates anonymized videos while preserving action recognition capabilities and evaluates privacy protection through advanced re-identification metrics.

## 📁 Project Structure

```
s2737744/
├── deep_privacy2/                # DeepPrivacy2 implementation
├── detectron2/                   # Detectron2 for pose detection
├── mmaction2/                    # MMAction2 for action recognition
├── yolov5/                       # YOLOv5 for person detection
├── scripts/
│   ├── re-id/                    # Re-identification evaluation pipeline
│   │   ├── videos_to_frames.py           # Video frame extraction
│   │   ├── crop_fullbody.py              # Person detection and background removal
│   │   ├── organize_reid_data.py         # Dataset organization for re-ID
│   │   ├── evaluate_reid_osnet.py        # Main privacy evaluation script
│   │   └── README.md                     # Pipeline documentation
│   ├── evaluation_face_iden.py   # Face identification evaluation
│   ├── evaluation_vid_quality.py # Video quality assessment
│   └── utility/                  # Utility scripts
├── datasets/                     # Dataset storage and processed data
│   ├── video_frames/             # Extracted video frames
│   ├── cropped/                  # Cropped person images
│   └── reid_eval_130frames/      # Re-ID evaluation data (121 frames)
├── output/                       # Anonymized video outputs
├── logs/                         # Evaluation logs
├── shared_cache/                 # Shared model cache
└── README.md                     # This file
```

## 🔐 Privacy Evaluation Results

### 🎉 Latest Re-ID Assessment (121 Frames)
```
Rank-1 Accuracy: 9.92%
mAP Score: 16.68%
Privacy Level: EXCELLENT ✅
```
## 🚀 Complete Pipeline Workflow

### 1. Video Frame Extraction
```bash
python scripts/re-id/videos_to_frames.py \
    --original_video /path/to/original.avi \
    --anonymized_video /path/to/anonymized.mp4 \
    --output_dir datasets/video_frames
```

### 2. Person Detection and Background Removal
```bash
# Process original frames
python scripts/re-id/crop_fullbody.py \
    --input_dir datasets/video_frames/video_name/original_frames \
    --output_dir datasets/cropped/original_cropped \
    --remove_background

# Process anonymized frames
python scripts/re-id/crop_fullbody.py \
    --input_dir datasets/video_frames/video_name/anonymized_frames \
    --output_dir datasets/cropped/anonymized_cropped \
    --remove_background
```

### 3. Re-ID Data Organization
```bash
python scripts/re-id/organize_reid_data.py \
    --original_cropped_dir datasets/cropped/original_cropped \
    --anonymized_cropped_dir datasets/cropped/anonymized_cropped \
    --output_dir datasets/reid_eval \
    --frames_per_person 130 \
    --use_multiple_frames
```

### 4. Privacy Evaluation
```bash
# Activate environment
conda activate dp2_eval

# Run evaluation
python scripts/re-id/evaluate_reid_osnet.py \
    --query_dir datasets/reid_eval/query \
    --gallery_dir datasets/reid_eval/gallery
```

## 🔧 Core Components

### DeepPrivacy2 Integration
- **Purpose**: Pose-conditioned GAN for video anonymization
- **Features**: Temporal consistency, multi-person support, high-quality generation
- **Status**: ✅ Fully implemented and operational

### Re-Identification Evaluation Pipeline
- **Purpose**: Measure identity leakage after anonymization
- **Models**: OSNet (Omni-Scale Network) for feature extraction
- **Methodology**: DeepPrivacy2 evaluation approach
- **Status**: ✅ Complete pipeline with excellent results

### Person Detection and Cropping
- **Purpose**: Extract person regions for re-ID evaluation
- **Features**: MediaPipe pose detection, background removal
- **Fallback**: OpenCV DNN models
- **Status**: ✅ Robust implementation with multiple detection methods

### Action Recognition (MMAction2)
- **Purpose**: Evaluate utility preservation
- **Framework**: MMAction2 for video understanding
- **Status**: ✅ Integrated for downstream task evaluation

## 📈 Research Contributions

This project makes significant contributions to privacy-preserving computer vision:

1. **Comprehensive Privacy Evaluation**: Novel 121-frame re-identification assessment
2. **Excellent Privacy Protection**: 9.92% Rank-1 accuracy demonstrates superior anonymization
3. **Temporal Consistency**: Smooth anonymized video generation with pose conditioning
4. **Multi-Modal Metrics**: Face detection, identity drift, and quality assessment
5. **Utility Preservation**: Action recognition evaluation for downstream tasks
6. **Real-World Applicability**: Practical implementation with state-of-the-art GANs

## 🎓 Academic Context

This work addresses the critical challenge of balancing privacy protection with data utility in video analysis. By implementing and evaluating pose-based anonymization, we contribute to:

- **Privacy-Preserving Computer Vision**: Novel evaluation methodologies with excellent results
- **Video Anonymization**: Practical implementation with DeepPrivacy2 GANs
- **Re-Identification Research**: Comprehensive privacy assessment frameworks
- **Temporal Consistency**: Ensuring smooth video anonymization

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd s2737744

# Create and activate environment
conda create -n dp2_eval python=3.8
conda activate dp2_eval

# Install dependencies (see individual component READMEs)
# - DeepPrivacy2: Follow deep_privacy2/README.md
# - Detectron2: Follow detectron2/README.md  
# - MMAction2: Follow mmaction2/README.md
# - TorchReID: pip install torchreid
```

## 📊 Evaluation Metrics

### Privacy Metrics
- **Rank-1 Accuracy**: Percentage of anonymized queries correctly matched to original identities
- **mAP**: Mean Average Precision across all ranks
- **Privacy Levels**: EXCELLENT/GOOD/MODERATE/POOR classification

### Quality Metrics
- **Face Detection Rate**: Measures privacy preservation (lower = better)
- **LPIPS**: Learned Perceptual Image Patch Similarity for identity drift
- **SSIM**: Structural Similarity Index for temporal consistency
- **Action Recognition Accuracy**: Measures utility preservation

## 🔍 DeepPrivacy2 Evaluation Approach

The pipeline follows the DeepPrivacy2 methodology:
- **Query Set**: Original frames (camera c1)
- **Gallery Set**: Anonymized frames (camera c2)
- **Goal**: Test if anonymized identities can be matched back to original identities
- **Success**: Lower re-identification rates = better anonymization

## 📝 License
This project is for academic research purposes. Please refer to individual component licenses for specific terms.

## 🤝 Acknowledgments
- **DeepPrivacy2**: [Original Implementation](https://github.com/hukkelas/deep_privacy2)
- **Detectron2**: Facebook Research
- **MMAction2**: OpenMMLab
- **OSNet**: [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
- **MediaPipe**: Google Research
