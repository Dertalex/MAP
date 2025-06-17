import src.generate_encodings as ge
import src.prediction_models as pm
from tqdm import tqdm
import os, sys
from joblib import parallel_backend
import ast
import sys
import math


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


import_data = "../../Data/HIS7_YEAST_Pokusaeva_2019.csv"
file_name = import_data.split("/")[-1].split(".")[0]

input_data = []
with open(import_data, "r") as infile:
    for line in infile.readlines():
        input_data.append(line[:-1].split(","))


import torch
torch.cuda.empty_cache()

to_encode = [line[1] for line in input_data[1:]]

e_type = sys.argv[1]
# e_type = "esm2_650M"

batch_size = 5000
# for e_type in ["esm2_650M", "esm1b"]:

print(f"now encoding {e_type}")

with (tqdm(total=len(to_encode)) as pbar):

    pad = 1+math.floor(math.log(batch_size, 10))
    for i in range(0, len(to_encode), batch_size):
        with HiddenPrints():
            encodings = ge.generate_sequence_encodings(e_type, to_encode[i:i+batch_size])
            outpath = f"../../Data/Embeddings/{file_name}/{e_type}/batch_{i+1:0{pad}d}"
            ge.save_encodings(encodings, outpath)
        pbar.update(batch_size)
