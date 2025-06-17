#!/bin/bash
#SBATCH --job-name="encode_esmc"
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --gpus=v100
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo $1 $2
python 03_encode_esmc.py $1 $2
