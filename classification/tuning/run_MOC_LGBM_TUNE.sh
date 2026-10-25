#!/bin/bash
#SBATCH --job-name=ml_tune             # Job name
#SBATCH --output=logs/tune_%j.log      # Output log file (use job ID)
#SBATCH --error=logs/tune_%j.err       # Error log file (use job ID)
#SBATCH --time=24:00:00                # Maximum runtime
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=32G                      # Memory allocation
#SBATCH --partition=gpu                # Partition (GPU queue)

# Load required modules
module purge
module load cuda/11.4

# Activate Conda environment
source ~/miniforge3/etc/profile.d/conda.sh  # Ensure Conda is available
conda activate cyber-risk   # name of your conda environment

# Run from the task directory (classification/ or regression/):  sbatch tuning/<this script>
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Run the tuning script
python tuning/MOC_LGBM_TUNE.py
