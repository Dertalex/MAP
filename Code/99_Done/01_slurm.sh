#!/bin/bash
#SBATCH --job-name="MAP_Benchmark"
#SBATCH --partition=paula
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load CUDAcore

python 01_MAP_cluster_test.py
