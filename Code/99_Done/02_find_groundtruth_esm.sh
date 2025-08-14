#!/bin/bash
#SBATCH --job-name="MAP_create_jobs"
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=main-%j.out
#SBATCH --error=main-%j.err
#SBATCH --mail-type=END,BEGIN,FAIL 

module load CUDAcore

# Define arrays


models=("xgboost" "xgboost_rf" "rf" "adaboost" "gboost")
iterations=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")


# Loop over e_types and models

for model in "${models[@]}"; do
  for iteration in "${iterations[@]}"; do 
    echo "$model"
    echo "$iteration"
    sbatch 02_find_groundtruth.sh "$model" "$iteration" 
    #./slurm_esm.sh "$e_type" "$model" "$iteration"
  done
done
