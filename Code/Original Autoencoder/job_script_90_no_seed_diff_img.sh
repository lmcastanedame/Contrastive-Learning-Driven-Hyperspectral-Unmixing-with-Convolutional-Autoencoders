#!/bin/bash
#SBATCH --job-name=90_noSeed_diff                   # Job name
#SBATCH --output=Batch/Output/output_%j.log         # Output file, %j will insert the job ID
#SBATCH --error=Batch/Error/error_%j.log            # Error file
#SBATCH --time=2-00:00:00                           # Maximum runtime
#SBATCH --partition=V100                            # Partition
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks (processes)
#SBATCH --gpus=2                                    # Number of GPUs (if needed)
#SBATCH --mem=32G                                   # Total memory

# Set CUDA-related environment variables
export TF_GPU_ALLOCATOR=cuda_malloc_async          # Optimized GPU memory allocation
export CUDA_VISIBLE_DEVICES=0,1,2                  # Explicitly specify GPUs to use

# Initialize Conda for this shell session
eval "$(/home/ids/castaneda-23/anaconda3/bin/conda shell.bash hook)"

# Activate the environment
conda activate CNNAEU

# Run your Python script
srun python3 'Method_CNNAEU_90_images (no seed, diff img).py'               