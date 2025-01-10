#!/bin/bash
#SBATCH --partition=a100               # GPU partition
#SBATCH --nodes=1                       # Use 1 node
#SBATCH --cpus-per-task=10             # Use 10 CPU cores
#SBATCH --gres=gpu:a100                 # Request 1 Tesla A100 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00                 # Maximum runtime
#SBATCH --job-name=t5_FineTune          # Job name
#SBATCH --output=logs/t5_output.log     # Standard output log
#SBATCH --error=logs/t5_error.log       # Standard error log


echo "Run started at $(date)"
echo "Environment variables:"
env
export HF_HOME=/home/woody/iwal/iwal196h/huggingface_cache
export TRANSFORMERS_CACHE=/home/woody/iwal/iwal196h/transformers_cache
export DATASETS_CACHE=/home/woody/iwal/iwal196h/datasets_cache
export PATH=/home/hpc/iwal/iwal196h/.conda/envs/text2sql/bin:$PATH


mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $DATASETS_CACHE
mkdir -p logs


export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


source ~/.bashrc
conda activate text2sql

python main.py

echo "Run completed at $(date)"
