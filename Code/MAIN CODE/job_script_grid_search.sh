#!/bin/bash
#SBATCH --job-name=grid_search                  # Job name
#SBATCH --output=Batch/Output/Grid_Search_%j.log      # Output file, %j will insert the job ID
#SBATCH --error=Batch/Error/Grid_Search__%j.log         # Error file
#SBATCH --time=2-00:00:00                          # Maximum runtime
#SBATCH --partition=V100                         # Partition
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks (processes)
#SBATCH --cpus-per-task=20                       # Number of CPU cores per task
#SBATCH --gpus=2                                 # Number of GPUs (if needed)
#SBATCH --mem=32G                                # Total memory

export TF_GPU_ALLOCATOR=cuda_malloc

# Initialize Conda for this shell session
eval "$(/home/ids/castaneda-23/anaconda3/bin/conda shell.bash hook)"

# Activate the environment
conda activate CNNAEU

# Run your Python script
srun python3 grid-search.py