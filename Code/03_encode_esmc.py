import os
import sys
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import math
import tqdm
from joblib import parallel_backend

dataset = sys.argv[1]
import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
method = sys.argv[2]

to_encode = []
with open(import_data, "r") as infile:
    for i, line in enumerate(infile.readlines()[1:]):
        line = line[:-1].split(",")
        sequence = line[1]
        score = round(float(line[2]), 3)
        to_encode.append((i, sequence, score))

pairings_file = f"../Data/Embeddings/{dataset}/pairings.csv"
with open(pairings_file, "w") as f:
    f.write(f"Embedding-Score-Pairings for {dataset}\n")

torch.cuda.empty_cache()
import gc

gc.collect()

# for method in (esmc_300m, esmc_600m)

pad = 1 + math.floor(math.log(len(to_encode), 10))
batch_size = 5000
with tqdm.tqdm(total=len(to_encode)) as pbar:
    for i in range(0, len(to_encode), batch_size):
        batch = to_encode[i:i + batch_size]

        for i, sequence, score in batch:
            protein = ESMProtein(sequence=sequence)
            client = ESMC.from_pretrained(method).to("cuda")  # or "cpu"
            protein_tensor = client.encode(protein)
            logits_output = client.logits(protein_tensor,
                                          LogitsConfig(sequence=True, return_embeddings=True,
                                                       return_hidden_states=True))

            mean_embeddings = torch.mean(logits_output.hidden_states, dim=-2)
            outfile = f"{i:0{pad}d}"

            for repr_layer in range(0,36):
                out_path = f"../Data/Embeddings/{dataset}/{method}/{repr_layer}/"
                if not os.path.exists(f"{out_path}"):
                    os.makedirs(f"{out_path}")

                representation = mean_embeddings[repr_layer, :]

                torch.save(representation,
                           os.path.join(out_path, outfile),
                           _use_new_zipfile_serialization=False)

            with open(pairings_file, "a") as f:
                f.write(f" {outfile},{score}\n")

            pbar.update(1)
