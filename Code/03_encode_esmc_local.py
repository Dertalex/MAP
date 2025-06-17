import os
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import math
import tqdm
import sys

# dataset = sys.argv[1]
# method = sys.argv[2]

dataset = "GRB2_HUMAN_Faure_2021"
method = "esmc_600m"

print(f"Script started to create {method} Embeddings for {dataset} sequences") 

# Create Pairings File to ensure that the embeddings are in the same order as the scores
pairings_csv = f"pairings.csv"
dirs = f"../Data/Embeddings/{dataset}"
pairings_file = os.path.join(dirs,pairings_csv)
# if not os.path.exists(pairings_file):
with open(pairings_file, "w") as f:
    f.write(f"Embedding-Score-Pairings for {dataset}\n")

import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
input_data = []
with open(import_data, "r") as infile:
    for line in infile.readlines():
        input_data.append(line[:-1].split(","))

to_encode = [line[1] for line in input_data[1:]]
scores = [line[2] for line in input_data[1:]]


import gc
gc.collect()
torch.cuda.empty_cache()

pad = 1 + math.floor(math.log(len(to_encode), 10))
with tqdm.tqdm(total=len(to_encode),desc=f"now creating {method} embeddings for {dataset}") as pbar:
    for i, sequence in enumerate(to_encode):
        # if method == "esmc_600m" and dataset == "GRB2_HUMAN_Faure_2021" and i < 39109:
        #     # print(i)
        #     pbar.update(1)
        #     continue
        # protein = ESMProtein(sequence=sequence)
        # client = ESMC.from_pretrained(method).to("cuda")  # or "cpu"
        # protein_tensor = client.encode(protein)
        # logits_output = client.logits(protein_tensor,
        #                             LogitsConfig(sequence=True, return_embeddings=True,
        #                                         return_hidden_states=True))
        #
        # mean_embeddings = torch.mean(logits_output.hidden_states, dim=-2)
        outfile = f"{i:0{pad}d}"

        # try:
        #     for repr_layer in range(0,99):
        #
        #         representation = mean_embeddings[repr_layer, :]
        #
        #         out_path = f"../Data/Embeddings/{dataset}/{method}/{repr_layer}/"
        #         if not os.path.exists(f"{out_path}"):
        #             os.makedirs(f"{out_path}")
        #
        #
        #         torch.save(representation,
        #                 os.path.join(out_path, outfile),
        #                 _use_new_zipfile_serialization=False)
        #
        # except IndexError:
        #     pass

        with open(pairings_file, "a") as f:
            f.write(f"{outfile},{scores[i]}\n")

        pbar.update(1)
