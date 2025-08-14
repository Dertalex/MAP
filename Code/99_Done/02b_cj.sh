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


embeddings=("blosum45" "blosum50" "blosum62" "blosum80" "blosum90" "georgiev")
models=("xgboost" "xgboost_rf" "rf" "adaboost" "lightgbm")
iterations=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")

#embeddings=("blosum45")
#models=("lightgbm")
#iterations=("1")


# Loop over e_types and models
for emb in "${embeddings[@]}"; do
  for model in "${models[@]}"; do
    for iteration in "${iterations[@]}"; do 
      sbatch 02b_find_gt_dual.sh "$emb" "$model" "$iteration" 
      #./02b_find_gt_dual.py "$embedding" "$model" "$iteration"
    done
  done
done
