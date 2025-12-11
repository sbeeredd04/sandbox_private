#!/bin/bash
# Distributed Training Script for ResNet18 on CelebA
# This script runs the training with Distributed Data Parallel (DDP)
# Using GPUs 1, 2, 3, 4 only

echo "=================================================="
echo "    Distributed Training for ResNet18 on CelebA"
echo "=================================================="
echo ""

# Check if CUDA is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Error: CUDA is not available!"
    exit 1
fi

# GPU configuration
GPU_IDS="1,2,3,4"
NUM_GPUS=4

echo "Using GPU IDs: $GPU_IDS"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Default parameters
EPOCHS=40
BATCH_SIZE=1024
LR=0.001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--epochs N] [--batch-size N] [--lr X] [--gpu-ids IDS]"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Total Batch Size: $BATCH_SIZE"
echo "  Batch Size per GPU: $((BATCH_SIZE / NUM_GPUS))"
echo "  Learning Rate: $LR"
echo ""

# Kill any existing training processes
echo "Checking for existing training processes..."
pkill -f train_resnet18_ddp.py 2>/dev/null && echo "  Killed existing training processes" || echo "  No existing processes found"
sleep 2

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('  GPU memory cleared')"

echo ""
echo "=================================================="
echo "Starting Distributed Training..."
echo "=================================================="
echo ""

# Run the training script
python train_resnet18_ddp.py --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --gpu-ids $GPU_IDS

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved as: best_celeba_resnet18_ddp.pth"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "=================================================="

exit $EXIT_CODE
