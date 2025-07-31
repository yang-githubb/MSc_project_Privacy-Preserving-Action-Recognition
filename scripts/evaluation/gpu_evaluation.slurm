#!/bin/bash
#SBATCH --job-name=reid_eval_gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=tc067-s2737744
#SBATCH --output=/work/tc067/tc067/s2737744/logs/eval/reid_eval_gpu_%A.out
#SBATCH --error=/work/tc067/tc067/s2737744/logs/eval/reid_eval_gpu_%A.err

# Initialize conda with the correct path
export PATH="/work/tc067/tc067/s2737744/miniconda3/bin:$PATH"
source /work/tc067/tc067/s2737744/miniconda3/etc/profile.d/conda.sh
conda activate dp2_eval

export IOPATH_CACHE=/work/tc067/tc067/s2737744/shared_cache/iopath
export FVCORE_CACHE=/work/tc067/tc067/s2737744/shared_cache/fvcore
export XDG_CACHE_HOME=/work/tc067/tc067/s2737744/shared_cache/xdg
export MPLCONFIGDIR=/work/tc067/tc067/s2737744/shared_cache/mpl
export PYTHONPATH=/work/tc067/tc067/s2737744/detectron2:$PYTHONPATH

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd /work/tc067/tc067/s2737744

# Create logs directory
mkdir -p logs/eval

echo "🚀 Starting GPU-only Re-ID evaluation"
echo "📅 Started at: $(date)"
echo "🔧 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "💾 Memory: $(free -h | grep Mem | awk '{print $2}')"

# Define paths
REID_OUTPUT="/work/tc067/tc067/s2737744/output/reid_evaluation"

# Check if preprocessing is complete
if [ ! -f "$REID_OUTPUT/preprocessing_complete.flag" ]; then
    echo "❌ Error: Preprocessing not complete. Please run the CPU preprocessing job first."
    echo "Run: sbatch scripts/preprocess_reid_cpu.slurm"
    exit 1
fi

echo "✅ Preprocessing complete, starting GPU evaluation..."

# Step 4: Run Re-ID evaluation (GPU optimized)
echo "🔍 Running Re-ID evaluation (GPU optimized)..."
python scripts/re-id/evaluate_reid_osnet.py \
    --query_dir "$REID_OUTPUT/reid_original" \
    --gallery_dir "$REID_OUTPUT/reid_anonymized" \
    --output_file "$REID_OUTPUT/reid_results.json" \
    --batch_size 64 \
    --num_workers 8 \
    --use_gpu \
    --mixed_precision

echo "✅ GPU Re-ID evaluation completed at: $(date)"
echo "📁 Results saved to: $REID_OUTPUT/reid_results.json"
echo "🔐 Privacy assessment completed successfully!"
echo "⏱️ Total GPU evaluation time: $SECONDS seconds" 