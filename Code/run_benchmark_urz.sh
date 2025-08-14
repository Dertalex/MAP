#!/bin/bash
#SBATCH --job-name=MAP_BENCH
#SBATCH --array=0-40319
#SBATCH --partition=paul
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --output=slurm_logs/MAP_BENCH_%a.out
#SBATCH --error=slurm_logs/MAP_BENCH_%a.err
mkdir -p slurm_logs

# Define arrays for parameters
iterations=(1 2 3 4 5 6 7 8 9 10)
datasets=("GFP" "PHOT" "RASK" "YAP1")
models=("linear" "ridge" "svr" "rf" "lightgbm" "xgboost")
embeddings=("one_hot" "georgiev" "blosum45" "blosum50" "blosum62" "blosum80" "blosum90")
n_startingpoints=(100 200 500 1000 2000 5000)
n_gain=(10 20 50 100)

# Get sizes
NUM_ITERATIONS=${#iterations[@]}
NUM_DATASETS=${#datasets[@]}
NUM_MODELS=${#models[@]}
NUM_EMBEDDINGS=${#embeddings[@]}
NUM_STARTINGPOINTS=${#n_startingpoints[@]}
NUM_GAIN=${#n_gain[@]}

SLURM_ID=${SLURM_JOB_ID}

#padded Array_ID (i.e. 1 of 100 -> 001)
padding_width=${#SLURM_ARRAY_TASK_COUNT}
ARRAY_ID=$(printf "%0*d" "$padding_width" "$SLURM_ARRAY_TASK_ID")

# Parameter indexing
ITERATION_INDEX=$(( ARRAY_ID % NUM_ITERATIONS ))
TEMP_ID=$(( ARRAY_ID / NUM_ITERATIONS ))

DATASET_INDEX=$(( TEMP_ID % NUM_DATASETS ))
TEMP_ID=$(( TEMP_ID / NUM_DATASETS ))

MODEL_INDEX=$(( TEMP_ID % NUM_MODELS ))
TEMP_ID=$(( TEMP_ID / NUM_MODELS ))

EMBEDDING_INDEX=$(( TEMP_ID % NUM_EMBEDDINGS ))
TEMP_ID=$(( TEMP_ID / NUM_EMBEDDINGS ))

STARTINGPOINTS_INDEX=$(( TEMP_ID % NUM_STARTINGPOINTS ))
TEMP_ID=$(( TEMP_ID / NUM_STARTINGPOINTS ))

GAIN_INDEX=$(( TEMP_ID % NUM_GAIN ))

# Assign variables
CURRENT_ITERATION=${iterations[ITERATION_INDEX]}
CURRENT_DATASET=${datasets[DATASET_INDEX]}
CURRENT_MODEL=${models[MODEL_INDEX]}
CURRENT_EMBEDDING=${embeddings[EMBEDDING_INDEX]}
CURRENT_STARTINGPOINTS=${n_startingpoints[STARTINGPOINTS_INDEX]}
CURRENT_GAIN=${n_gain[GAIN_INDEX]}

# Debug info
#echo "Starting task ${SLURM_ARRAY_TASK_ID} with:"
#echo "  Iteration: ${CURRENT_ITERATION}"
#echo "  Dataset: ${CURRENT_DATASET}"
#echo "  Embedding: ${CURRENT_EMBEDDING}"
#echo "  Model: ${CURRENT_MODEL}"
#echo "  Starting Points: ${CURRENT_STARTINGPOINTS}"
#echo "  Gain: ${CURRENT_GAIN}"

# Call Python script
python 12_MAP_Benchmark_ZS_setup.py "${CURRENT_ITERATION}" "${CURRENT_DATASET}" "${CURRENT_EMBEDDING}" "${CURRENT_MODEL}" "${CURRENT_STARTINGPOINTS}" "${CURRENT_GAIN}" "${ARRAY_ID}"
