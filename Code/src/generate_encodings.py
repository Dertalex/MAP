import math
import os
import warnings
import numpy as np
from typing import Literal, Optional
import src.Blosum as bl
import src.georgiev_parameters as gg
from torch_geometric.data import Data
from Bio.PDB import PDBParser

''' structural graph encoding'''

def generate_graph_encoding(pdb_file, 
                            y, 
                            node_features: Literal["one_hot", "georgiev", "blosum45", "blosum50",
                                            "blosum62", "blosum80", "blosum90"], 
                            distance_threshold: float = 8.0) -> Data:

    import torch
    from torch_geometric.data import Data
    from Bio.PDB import PDBParser

    aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    aa3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
              'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
              'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    aa_to_index = {aa: i for i, aa in enumerate(aa_codes)}


    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = []
    features = []
    res_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    res_coords.append(residue["CA"].coord)

                    #update features for every residue/node
                    f_vector = []
                    if node_features == "one_hot":
                        f_vector = np.zeros(len(aa_codes))
                        f_vector[aa_to_index[residue.get_resname()]] = 1

                    elif node_features == "georgiev":
                        for parameter in gg.GEORGIEV_PARAMETERS:
                            f_vector.append(parameter[aa3to1[residue.get_resname()]])

                    elif node_features in ["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"]:
                        bl_matrices = [bl.blosum_45, bl.blosum_50, bl.blosum_62, bl.blosum_80, bl.blosum_90]
                        blosum = bl_matrices[["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"].index(features)]
                        for aa in aa_codes:
                            f_vector.append(blosum[aa3to1[residue.get_resname()]][aa3to1[aa]])

                    features.append(f_vector)
                else:
                    raise ValueError(f"Invalid residue: {residue.get_resname()}")
                
    # Convert coordinates and features to tensors
    coords_tensor = torch.tensor(res_coords, dtype=torch.float)
    features = np.array(features)
    features_tensor = torch.tensor(features, dtype=torch.float)

    # Create edges
    edge_indeces = []
    num_nodes = int(coords_tensor.shape[0])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.norm(coords_tensor[i] - coords_tensor[j]) < distance_threshold:
                edge_indeces.append([i, j])
                edge_indeces.append([j, i])

    edges = torch.tensor(edge_indeces, dtype=torch.long).t().contiguous()

    return Data(x=features_tensor, edge_index=edges, y=y)



''' sequence representation encodings'''


def generate_sequence_encodings(method: Literal[
    "one_hot", "georgiev", "blosum45", "blosum50", "blosum62", "blosum80",
    "blosum90", "esmc_300m", "esmc_600m", "prost_t5"],
                                sequences: list, esm_batch_size: Optional[int] = None) -> list:
    """
    create one hot encodings from AA sequences. Size of OHE and BLOSUM tensors is determined by the longest sequence in the list.


    """

    # define default tensor length as the length of the longest sequence
    tensor_length = max([len(sequence) for sequence in sequences])

    """ georgiev encoding """

    if method == "georgiev":
        encodings = []
        for sequence in sequences:
            sequence = list(sequence)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            encoding = []
            for aa in sequence:
                gg_vector = []
                for parameter in gg.GEORGIEV_PARAMETERS:
                    gg_vector.append(parameter[aa])
                encoding.append(gg_vector)
            encodings.append(np.array(encoding))

        return encodings
    """ one hot encoding """

    if method == "one_hot":
        encodings = []
        for sequence in sequences:
            sequence = list(sequence)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            aa_enumerated = {aa: i for i, aa in enumerate(amino_acids)}

            # encoding = [[0 for j in range(20)] for j in range(tensor_length)]  # list of lists format
            encoding = np.zeros((len(sequence), 20))  # numpy array format
            i = 0
            for aa in sequence:
                encoding[i][aa_enumerated[aa]] = 1
                i += 1
            encodings.append(np.array(encoding))

        return encodings

    """ blosum encodings """

    if method in ["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"]:

        bl_matrices = [bl.blosum_45, bl.blosum_50, bl.blosum_62, bl.blosum_80, bl.blosum_90]
        blosum = bl_matrices[["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"].index(method)]

        encodings = []
        for sequence in sequences:
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            aa_enumerated = {i: aa for i, aa in enumerate(amino_acids)}

            # encoding = [[0 for j in range(20)] for j in range(tensor_length)]  # list of lists format
            encoding = np.zeros((len(sequence), 20))  # numpy array format

            i = 0
            for aa_seq in sequence:

                for aa_num in aa_enumerated:
                    encoding[i][aa_num] = blosum[aa_seq][aa_enumerated[aa_num]]
                i += 1
            encodings.append(np.array(encoding))

        return encodings

    ''' ESM encodings '''

    if method in ["esm1b", "esm2_650M", "esm2_8M", "esm2_3B"]:
        sequences = [(i, sequence) for i, sequence in enumerate(sequences)]

        import torch
        import esm

        if method == "esm1b":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        elif method == "esm2_650M":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        elif method == "esm2_8M":
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        else:  # method == "esm_3B"
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disable dropout for deterministic output

        # detect device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise Warning("No GPU available, you really don't want to use this on CPU")

        r_layers = {"esm1b": 33, "esm2_650M": 33, "esm2_8M": 6, "esm2_3B": 36}
        model = model.to(device)

        if esm_batch_size is None:
            batchsize = 1
        else:
            batchsize = esm_batch_size
        encodings = []

        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disable dropout for deterministic output
        # detect device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise Warning("No GPU available, you really don't want to use this on CPU")
        model = model.to(device)
        batchsize = 1
        encodings = []
        for i in range(0, len(sequences), batchsize):
            batch = sequences[i: i + batchsize]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[r_layers[method]], return_contacts=True)
            token_representations = results["representations"][r_layers[method]]
            for j, tokens_len in enumerate(batch_lens):
                encodings.append(token_representations[j, 1: tokens_len - 1].mean(0))
        return encodings

    if method in ["esmc_300m", "esmc_600m"]:
        import torch
        import tqdm
        from transformers import AutoModelForMaskedLM

        representations = []

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

            with tqdm.tqdm(total=len(sequences)) as pbar:
                for sequence in sequences:

                    with torch.no_grad():
                        tokenized = tokenizer(sequence, return_tensors="pt").to(device)
                        output = model(**tokenized, output_hidden_states=True)

                    token_embeddings = output.hidden_states  # tuple: one tensor per layer

                    for j, layer in enumerate(token_embeddings): # shape: (batch, seq_len+2, hidden_dim)
                        residue_embeddings = layer[0, 1:len(sequence)+1, :] # take only amino acids, exclude CLS (0) and EOS (-1)
                        sequence_embedding = residue_embeddings.mean(0)  # (hidden_dim,)
                    representations.append(sequence_embedding.cpu())
                    pbar.update(1)

            return representations


    """ProtTrans Encodings"""
    if method in ["protT5"]:
        from transformers import T5Tokenizer, T5EncoderModel
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            warnings.warn("No GPU available, using CPU (this will be slow)")

        # Load model + tokenizer
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
        model.eval()

        representations = []
        
        # Process sequences
        for sequence in sequences:
            # Prepare sequence (add spaces between amino acids)
            prepped_seq = " ".join(list(sequence))
            
            # Tokenize
            ids = tokenizer.batch_encode_plus(
                [prepped_seq],
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt"
            )
            
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)
            
            # Extract embeddings
            with torch.no_grad():
                embedding_output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            residue_embs = embedding_output.last_hidden_state
            
            # Get valid indices (excluding padding and special tokens)
            valid_indices = torch.where(attention_mask[0, 1:] == 1)[0] + 1
            
            if valid_indices.numel() > 0:
                # Mean pool over valid residues
                seq_embedding = residue_embs[0, valid_indices].mean(dim=0).cpu()
            else:
                warnings.warn(f"Empty sequence tokenization for sequence: {sequence[:20]}...")
                seq_embedding = torch.zeros(residue_embs.shape[2], dtype=torch.float32)
            
            encodings.append(seq_embedding)
        
        return encodings

def load_encodings(encodings_folder):
    """loads all MAP encodings from a folder"""
    encodings = []
    is_torch = False
    saved_encodings = os.listdir(encodings_folder)
    saved_encodings = sorted(saved_encodings)
    if saved_encodings[1].endswith(".npy"):
        for saved_encoding in saved_encodings:
            encodings.append(np.load(os.path.join(encodings_folder, saved_encoding)))
    else:
        import torch
        for saved_encoding in saved_encodings:
            encodings.append(torch.load(os.path.join(encodings_folder, saved_encoding)))

    return encodings


if __name__ == '__main__':
    # print(generate_sequence_encodings(method="one_hot", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
    # print(generate_sequence_encodings(method="blosum62", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
    encodings = generate_sequence_encodings(method="georgiev",
                                            sequences=["AAACDEFGHIKLMNPQRSTVWY", "AAACDEFGHIMLKMNPQRSTVWY"])

    # print(generate_sequence_encodings(method="esm2", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))RSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
