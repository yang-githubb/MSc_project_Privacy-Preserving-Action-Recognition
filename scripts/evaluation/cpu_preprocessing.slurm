#!/bin/bash
#SBATCH --job-name=reid_preprocess
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/eval/reid_preprocess_%A.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/eval/reid_preprocess_%A.err

# Initialize conda with the correct path
export PATH="/work/tc067/tc067/s2737744/miniconda3/bin:$PATH"
source /work/tc067/tc067/s2737744/miniconda3/etc/profile.d/conda.sh
conda activate dp2_eval

export IOPATH_CACHE=/work/tc067/tc067/s2737744/shared_cache/iopath
export FVCORE_CACHE=/work/tc067/tc067/s2737744/shared_cache/fvcore
export XDG_CACHE_HOME=/work/tc067/tc067/s2737744/shared_cache/xdg
export MPLCONFIGDIR=/work/tc067/tc067/s2737744/shared_cache/mpl
export PYTHONPATH=/work/tc067/tc067/s2737744/detectron2:$PYTHONPATH

# CPU optimization settings
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

cd /work/tc067/tc067/s2737744

# Create logs directory
mkdir -p logs/eval

echo "🖥️ Starting CPU-only Re-ID preprocessing"
echo "📅 Started at: $(date)"
echo "💻 CPU cores: $(nproc)"
echo "💾 Memory: $(free -h | grep Mem | awk '{print $2}')"

# Define paths
ORIGINAL_VIDEOS="/work/tc067/tc067/s2737744/Dataset/ucf101/UCF-101"
ANONYMIZED_VIDEOS="/work/tc067/tc067/s2737744/output/ucf101_anonymized"
REID_OUTPUT="/work/tc067/tc067/s2737744/output/reid_evaluation"

# Create Re-ID evaluation output directory
mkdir -p "$REID_OUTPUT"

echo "📊 Running CPU preprocessing for Re-ID evaluation..."
echo "Original videos: $ORIGINAL_VIDEOS"
echo "Anonymized videos: $ANONYMIZED_VIDEOS"
echo "Re-ID output: $REID_OUTPUT"

# Step 1: Extract frames from videos (CPU optimized)
echo "📹 Step 1: Extracting frames from videos (CPU optimized)..."
python scripts/re-id/videos_to_frames.py \
    --input_dir "$ORIGINAL_VIDEOS" \
    --output_dir "$REID_OUTPUT/original_frames" \
    --frame_interval 5 \
    --num_workers 16 \
    --skip_existing

python scripts/re-id/videos_to_frames.py \
    --input_dir "$ANONYMIZED_VIDEOS" \
    --output_dir "$REID_OUTPUT/anonymized_frames" \
    --frame_interval 5 \
    --num_workers 16 \
    --skip_existing

# Step 2: Extract person crops (CPU optimized)
echo "👤 Step 2: Extracting person crops (CPU optimized)..."
python scripts/re-id/crop_fullbody.py \
    --input_dir "$REID_OUTPUT/original_frames" \
    --output_dir "$REID_OUTPUT/original_crops" \
    --num_workers 16 \
    --skip_existing

python scripts/re-id/crop_fullbody.py \
    --input_dir "$REID_OUTPUT/anonymized_frames" \
    --output_dir "$REID_OUTPUT/anonymized_crops" \
    --num_workers 16 \
    --skip_existing

# Step 3: Organize Re-ID dataset (CPU optimized)
echo "📁 Step 3: Organizing Re-ID dataset (CPU optimized)..."
python scripts/re-id/organize_reid_data.py \
    --input_dir "$REID_OUTPUT/original_crops" \
    --output_dir "$REID_OUTPUT/reid_original" \
    --num_workers 16

python scripts/re-id/organize_reid_data.py \
    --input_dir "$REID_OUTPUT/anonymized_crops" \
    --output_dir "$REID_OUTPUT/reid_anonymized" \
    --num_workers 16

echo "✅ CPU preprocessing completed at: $(date)"
echo "📁 Preprocessed data ready for GPU evaluation"
echo "⏱️ Total preprocessing time: $SECONDS seconds"

# Create a flag file to indicate preprocessing is complete
touch "$REID_OUTPUT/preprocessing_complete.flag"
echo "🚀 Ready for GPU evaluation job!" 