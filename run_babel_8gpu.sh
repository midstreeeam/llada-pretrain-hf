#!/bin/bash
#SBATCH --job-name=llada_135m_bb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=0  # Use all available memory on the node

# Load necessary modules (if any, though user environment seems to be handled via conda/venv)
# module load cuda/11.8  # Example, adjust based on cluster availability

# Activate virtual environment
source .venv/bin/activate  # Adjust path to your venv

# Set environment variables
export OMP_NUM_THREADS=16
export HF_HOME=/data/user_data/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/data/user_data/$USER/huggingface_cache
export HF_DATASETS_CACHE=/data/user_data/$USER/huggingface_cache

# Create log directory
mkdir -p logs

# Run training
torchrun --nproc_per_node=8 training_launcher.py \
    --config training_config/llada_135m_bb.yaml \
    --repo-dir .
