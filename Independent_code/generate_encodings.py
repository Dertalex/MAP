import numpy as np
from typing import Literal
import Blosum as bl
import georgiev_parameters as gg

def load_dataset():
    pass


''' sequence representation encodings'''


def generate_sequence_encodings(method: Literal["onehot", "georgiev",
Literal["blosum45", "blosum50", "blosum62", "blosum80", "blosum90"], "esm1b", "esm2"], sequences: list) -> list:
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

    if method == "onehot":
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
            encodings.append(encoding)

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
            encodings.append(encoding)

        return encodings

    ''' ESM encodings '''

    if method in ["esm1b", "esm2"]:
        sequences = [(i, sequence) for i, sequence in enumerate(sequences)]

        import torch
        import esm
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() if method == "esm1b" else esm.pretrained.esm2_t33_650M_UR50D()
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
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)

            token_representations = results["representations"][33]

            for j, tokens_len in enumerate(batch_lens):
                encodings.append(token_representations[j, 1: tokens_len - 1].mean(0))

        return encodings


def save_encoding():
    pass


def main():
    pass


if __name__ == '__main__':
    # print(generate_sequence_encodings(method="one_hot", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
    # print(generate_sequence_encodings(method="blosum62", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
    encodings = generate_sequence_encodings(method="georgiev",
                                          sequences=["AAACDEFGHIKLMNPQRSTVWY", "AAACDEFGHIMLKMNPQRSTVWY"])

    print(encodings[0].shape)
    print(encodings[0].__name__)
    # print(generate_sequence_encodings(method="esm2", sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIMLKMNPQRSTVWY"]))
