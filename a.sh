#!/bin/bash
#SBATCH --job-name=gpu_job          # Job name
#SBATCH --partition=gpu             # GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=64GB                   # Memory per node 
#SBATCH --time=2:00:00              # Time limit hrs:min:sec

module load cuda/12.4              # Load necessary modules (example)
srun python graphsnoninteractive.py        # Run your script
