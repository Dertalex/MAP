#!/bin/bash
#SBATCH --job-name="MAP_Opt"
#SBATCH --partition=paul
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo $1 $2 $3 $4
python 04_hyper_optuna_esm.py $1 $2 $3 $4
