import numpy as np
from typing import Literal
import Blosum as bl


def load_dataset():
    pass

''' sequence representation encodings'''

def generate_sequence_encodings(method: Literal["one_hot", 
                                         Literal["blosum45","blosum50","blosum62","blosum80","blosum90"],"esm1b", "esm2"], sequences: list) -> list:

    """
    create one hot encodings from AA sequences. Size of OHE and BLOSUM tensors is determined by the longest sequence in the list.


    """

    # define default tensor length as the length of the longest sequence
    tensor_length = max([len(sequence) for sequence in sequences])

    if method == "one_hot":
        encodings = []
        for sequence in sequences:
            sequence = list(sequence)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            aa_enumerated = {aa: i for i, aa in enumerate(amino_acids)}

            encoding = [[0 for j in range(20)] for j in range(tensor_length)]  # list of lists format
            # encoding = np.zeros((len(sequence), 20)) #numpy array format
            i = 0
            for aa in sequence:
                encoding[i][aa_enumerated[aa]] = 1
                i += 1
            encodings.append(encoding)

        return encodings

    bl_methods = ["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"]
    if method in bl_methods:
    
        bl_matrices = [bl.blosum_45, bl.blosum_50, bl.blosum_62, bl.blosum_80, bl.blosum_90]
        blosum = bl_matrices[bl_methods.index(method)]
        
        encodings = []       
        for sequence in sequences:
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            aa_enumerated = {i: aa for i, aa in enumerate(amino_acids)}

            encoding = [[0 for j in range(20)] for j in range(tensor_length)]  # list of lists format
            # encoding = np.zeros((len(sequence), 20))                          # numpy array format

            i = 0
            for aa_seq in sequence:

                for aa_num in aa_enumerated:
                    encoding[i][aa_num] = blosum[aa_seq][aa_enumerated[aa_num]]
                i += 1
            encodings.append(encoding)
        
        return encodings


''' protein structure encodings'''

def generate_esm_v1b_encoding():
    import torch
    import esm
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

def generate_esm2_encoding():
    pass


def save_encoding():
    pass


def main():
    pass


if __name__ == '__main__':
    print(generate_sequence_encodings(method="one_hot", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
    print(generate_sequence_encodings(method="blosum62", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))

