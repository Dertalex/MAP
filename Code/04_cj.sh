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

datasets=("HIS7_YEAST_Pokusaeva_2019" "CAPSD_AAV2S_Sinai_2021" "D7PM05_CLYGR_Somermeyer_2022" "GFP_AEQVI_Sarkisyan_2016" "GRB2_HUMAN_Faure_2021")
embeddings=("one_hot" "blosum45" "blosum50" "blosum62" "blosum80" "blosum90" "georgiev")
models=("xgboost" "rf" "adaboost" "lightgbm")
iterations=("1" "2" "3")
trials= 500

#embeddings=("blosum45")
#models=("lightgbm")
#iterations=("1")


# Loop over e_types and models
for dataset in "${datasets[@]}"; do
  for emb in "${embeddings[@]}"; do
    for model in "${models[@]}"; do
      for iteration in "${iterations[@]}"; do 
        sbatch 04_hypertune.sh "$dataset" "$emb" "$model" "$iteration" "$trials" 
        #./02b_find_gt_dual.py "$embedding" "$model" "$iteration"
      done
    done
  done
done
