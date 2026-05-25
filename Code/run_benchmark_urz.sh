#!/bin/bash

# env to be activated

# Define arrays for parameters
iterations=(1)
datasets=("YAP1")
models=("linear")
embeddings=("one_hot")
n_startingpoints=(100)
n_gain=(10)

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
ARRAY_ID=0

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
python 12_MAP_Benchmark_ZS_setup.py \
    "${CURRENT_ITERATION}" \
    "${CURRENT_DATASET}" \
    "${CURRENT_EMBEDDING}" \
    "${CURRENT_MODEL}" \
    "${CURRENT_STARTINGPOINTS}" \
    "${CURRENT_GAIN}" \
    "${ARRAY_ID}"
