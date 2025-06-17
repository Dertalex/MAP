#!/bin/bash
#SBATCH --job-name="Ground_Truth"
#SBATCH --partition=paul
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=find_gt_dual%j_$1_$2.out
#SBATCH --error=find_gt_dual%j_$1_$2.err
#SBATCH --mail-type=END,BEGIN,FAIL 

echo "$1"
echo "dual with OHE"
echo "$2"
echo "$3"

python 02_find_gt_dual.py $1 $2 $3
