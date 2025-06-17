#!/bin/bash
#SBATCH --job-name="encode_esmc"
#SBATCH --partition=paula
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G
#SBATCH --gpus=a30:1
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load CUDAcore/11.5.1
module load pytorch

echo $1
python encode_esmc.py $1
