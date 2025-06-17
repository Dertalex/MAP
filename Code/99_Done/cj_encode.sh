#!/bin/bash
#SBATCH --job-name="Create Encoding Jobs"
#SBATCH --partition=paul
#SBATCH --time=2-00:00:00
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=s%j_$1_$2.out
#SBATCH --error=s%j_$1_$2.err
#SBATCH --mail-type=END,BEGIN

# Define arrays
e_types=("esmc_300m" "esmc_600m")

for method in "${e_types[@]}"; do
  echo "$method" 
  sbatch encode_paula.sh "$method"
done
