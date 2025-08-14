#!/bin/bash
#SBATCH --job-name="Create Jobs"
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

# Define arrays
e_types=("one_hot" "blosum45" "blosum50" "blosum62" "blosum80" "blosum90" "georgiev")
models=("xgboost" "xgboost_rf" "rf" "gboost")

#e_types=("one_hot")
#models=("rf")

# Loop over e_types and models
for e_type in "${e_types[@]}"; do
  for model in "${models[@]}"; do

    echo "$e_type" 
    echo "$model" 
    sbatch 02_find_groundtruth.sh "$e_type" "$model" 
    #python 02_find_groundtruth.py "$e_type" "$model" 
  done
done
