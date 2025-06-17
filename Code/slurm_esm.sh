#!/bin/bash
#SBATCH --job-name="MAP_GT_ESM1b"
#SBATCH --partition=paul
#SBATCH --time=2-00:00:00
#SBATCH --mem=90G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo $1 $2
python 02_find_groundtruth_esm.py $1 $2
