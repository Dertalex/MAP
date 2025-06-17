#!/bin/bash

datasets=("GRB2_HUMAN_Faure_2021")
embeddings=("esmc_600m")

# Loop over e_types and models
for emb in "${embeddings[@]}"; do
  for dataset in "${datasets[@]}"; do
    python 03_encode_esmc_local.py "$dataset" "$emb"
  done
done
