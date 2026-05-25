import os
import torch
import tqdm
from transformers import AutoModelForMaskedLM

for dataset in ["RASK", "YAP1", "GFP", "PHOT"][:3]:  # first 3 datasets
    
    # Load sequences
    import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
    to_encode = []
    with open(import_data, "r") as infile:
        for line in infile.readlines()[1:]:
            mutant, sequence = line.strip().split(",")[:2]
            to_encode.append((mutant, sequence))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for method in ["esmc_600m", "esmc_300m"]:
        if method == "esmc_600m":
            model_name = "Synthyra/ESMplusplus_large"
        else:
            model_name = "Synthyra/ESMplusplus_small"

        # Load model + tokenizer
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        tokenizer = model.tokenizer
        model.eval()

        with tqdm.tqdm(total=len(to_encode), desc=f"Now creating {method} embeddings for {dataset}") as pbar:
            for mutant, sequence in to_encode:

                with torch.no_grad():
                    tokenized = tokenizer(sequence, return_tensors="pt").to(device)
                    output = model(**tokenized, output_hidden_states=True)

                token_embeddings = output.hidden_states  # tuple: one tensor per layer

                for j, layer in enumerate(token_embeddings): # shape: (batch, seq_len+2, hidden_dim)
                    residue_embeddings = layer[0, 1:len(sequence)+1, :] # take only amino acids, exclude CLS (0) and EOS (-1)
                    sequence_embedding = residue_embeddings.mean(0)  # (hidden_dim,)

                    out_path = f"../Data/Embeddings/{dataset}/{method}/{j}/"
                    os.makedirs(out_path, exist_ok=True)
                    torch.save(sequence_embedding, os.path.join(out_path, f"{mutant}"),
                               _use_new_zipfile_serialization=False)

                pbar.update(1)
