import math
import os
import warnings

import numpy as np
from typing import Literal, Optional
import src.Blosum as bl
import src.georgiev_parameters as gg


''' structural graph encoding'''

def generate_graph_encoding(features: Literal[
    "one_hot", "georgiev", "blosum45", "blosum50", "blosum62", "blosum80",
    "blosum90"]):
    from Bio.PDB import PDBParser, is_aa
    import torch
    from torch_geometric.data import Data

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", "example.pdb")

    model = structure[0]
    chain = next(model.get_chains())

    aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    aa3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
              'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
              'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    aa_to_index = {aa: i for i, aa in enumerate(aa_codes)}

    residues = []
    features = []
    res_coords = []

    for idx, res in enumerate(chain):
        if is_aa(res) and "CA" in res:
            res_coords.append(res["CA"].coord)
            f_vector = []
            if features == "one_hot":
                f_vector = np.zeros(len(aa_codes))
                f_vector[aa_to_index[res.get_resname()]] = 1

            elif features == "georgiev":
                for parameter in gg.GEORGIEV_PARAMETERS:
                    f_vector.append(parameter[res.get_resname()])

            elif features in ["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"]:
                bl_matrices = [bl.blosum_45, bl.blosum_50, bl.blosum_62, bl.blosum_80, bl.blosum_90]
                blosum = bl_matrices[["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"].index(features)]
                for aa in aa_codes:
                    f_vector.append(blosum[aa3to1[res.get_resname()]][aa3to1[aa]])

            features.append(f_vector)


''' sequence representation encodings'''


def generate_sequence_encodings(method: Literal[
    "one_hot", "georgiev", "blosum45", "blosum50", "blosum62", "blosum80",
    "blosum90", "esmc_300m", "esmc_600m"],
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

    if method in ["esmc_300m", "esmc_600m"]:
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        repr_layer = -1
        representations = []

        client = ESMC.from_pretrained(method).to("cuda")

        for sequence in sequences:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(protein_tensor,
                                          LogitsConfig(sequence=True, return_embeddings=True,
                                                       return_hidden_states=True))

            mean_embeddings = torch.mean(logits_output.hidden_states, dim=-2)
            representation = mean_embeddings[repr_layer, :]
            representations.append(representation)

        return representations

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

    """ProtTrans Encodings"""
    if method in ["prostT5", "protT5", "protT5_XL"]:

        from transformers import T5Tokenizer, T5EncoderModel
        import torch
        import re

        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise Warning("No GPU available, you really don't want to use this on CPU")

        if method == "prostT5":
            tokenizer = T5Tokenizer.from_pretrained("'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False")
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

        if method == "proT5_XL":
            tokenizer = T5Tokenizer.from_pretrained("'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False")
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

        if method == "protT5":
            tokenizer = T5Tokenizer.from_pretrained("'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False")
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)


def save_encodings(encodings, outpath):
    is_torch = False
    is_ndarr = False

    if encodings is None or len(encodings) == 0:
        print("no encodings provided")

    if not isinstance(encodings[0], np.ndarray):
        import torch
        if not isinstance(encodings[0], torch.Tensor):
            warnings.warn("provided encodings are not in numpy array or torch.Tensor format.")
        else:
            is_torch = True
    else:
        is_ndarr = True

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for i, encoding in enumerate(encodings):
        log_floor = math.floor(math.log(len(encodings), 10))
        outfile = os.path.join(outpath, f"{i:0{1 + log_floor}d}")
        if is_ndarr:
            np.save(outfile, encoding)
        if is_torch:
            torch.save(encoding, outfile)
    # print(f"Encodings saved to {outpath}")


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
