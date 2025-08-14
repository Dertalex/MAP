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

datasets=("CAPSD_AAV2S_Sinai_2021" "D7PM05_CLYGR_Somermeyer_2022" "GFP_AEQVI_Sarkisyan_2016" "GRB2_HUMAN_Faure_2021")
embeddings=("esmc_300m" "esmc_600m")

# Loop over e_types and models
for dataset in "${datasets[@]}"; do
  for emb in "${embeddings[@]}"; do
    sbatch 03_encode_esmc.sh "$dataset" "$emb"
  done
done
