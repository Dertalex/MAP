#!/bin/bash

LOG_DIR="../logs/logs_otf"
mkdir -p ../logs/err_logs_otf
mkdir -p "${LOG_DIR}"

# Parameters
iterations=(1 2 3 4 5 6 7 8 9 10)
datasets=("RASK")
models=("rf")
embeddings=("one_hot")
n_startingpoints=(1000)
n_gain=(100)

RESULTS_DIR="../Results/MLDE_Benchmark_otf"

RUN_ID=0

# Loop over all combinations
for CURRENT_ITERATION in "${iterations[@]}"; do
  for CURRENT_DATASET in "${datasets[@]}"; do
    for CURRENT_MODEL in "${models[@]}"; do
      for CURRENT_EMBEDDING in "${embeddings[@]}"; do
        for CURRENT_STARTINGPOINTS in "${n_startingpoints[@]}"; do
          for CURRENT_GAIN in "${n_gain[@]}"; do

            ARRAY_ID=$(printf "%03d" ${RUN_ID})

            echo "Running job ${ARRAY_ID} with:"
            echo "  Iteration: ${CURRENT_ITERATION}"
            echo "  Dataset: ${CURRENT_DATASET}"
            echo "  Embedding: ${CURRENT_EMBEDDING}"
            echo "  Model: ${CURRENT_MODEL}"
            echo "  Starting Points: ${CURRENT_STARTINGPOINTS}"
            echo "  Gain: ${CURRENT_GAIN}"

            python 12_MAP_Benchmark_ZS_setup.py \
              "${CURRENT_ITERATION}" \
              "${CURRENT_DATASET}" \
              "${CURRENT_EMBEDDING}" \
              "${CURRENT_MODEL}" \
              "${CURRENT_STARTINGPOINTS}" \
              "${CURRENT_GAIN}" \
              "${ARRAY_ID}" \
              "${RESULTS_DIR}" \
              "${LOG_DIR}"

            ((RUN_ID++))

          done
        done
      done
    done
  done
done
