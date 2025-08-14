#!/bin/bash
#SBATCH --job-name="Ground_Truth"
#SBATCH --partition=paul
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=s%j_$1_$2.out
#SBATCH --error=s%j_$1_$2.err
#SBATCH --mail-type=END,BEGIN,FAIL 

module load CUDAcore

python 02_find_groundtruth.py $1 $2
