import random
import sys, os
import warnings
from datetime import datetime
import math
import copy
import numpy as np
import torch
import inspect
from typing import Literal, Optional
from src.generate_encodings import generate_sequence_encodings



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HiddenWarnings():
    def __enter__(self):
        # Save the current filter settings before changing them
        self._previous_filters = warnings.filters[:]
        # Ignore all warnings
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original warning filter settings
        warnings.filters = self._previous_filters

class Protein():
    _id = str()
    _seq = str()
    _score = float()
    _zs_score = float()

    def __init__(self, id: str, sequence: str, score: float, zs_score: float):
        self._id = id
        self._seq = sequence
        self._score = score
        self._zs_score = zs_score
    
    def determine_wt(self):
        wild_type = self._seq
        mutations = self._id.split(":")
        for mut in mutations:
            index = int(mut[1:-1]) - 1
            wild_type = wild_type[:index] + mut[0] + wild_type[index + 1:]
        return wild_type


class Library(): #class for holding proteins. Used for representation Generation for cleaner code and better memory handling
    _wildtype_seq = str()
    _mutant_catalog = list[Protein]

    def __init__(self, mutant_ids: list[str] = None):
        self._mutant_catalog = mutant_ids if mutant_ids is not None else []

    def add_mutants(self, mutants: list[Protein]):
        for mutant in mutants:
            if mutant not in self._mutant_catalog:
                self._mutant_catalog.append(mutant)

    def remove_mutants(self, mutants: list[Protein]):
        for mutant in mutants:
            self._mutant_catalog.remove(mutant)

    def load_embeddings(self, path_to_mutants, device : Literal["cpu","cuda"] = "cpu"):
        
        if len(self._mutant_catalog) ==0:
            raise ValueError("Library is empty. Cannot load or generate mutants")

        if not os.path.exists(path_to_mutants):
            raise FileNotFoundError("Could not find Path to mutants")
        
        representations = []    
        for mutant in self._mutant_catalog:
            representations.append(torch.load(os.path.join(path_to_mutants,f"{mutant._id}.pt")),map_location=torch.device(device))
        
        return representations

    def generate_embeddings(self, type : Literal["one_hot", "georgiev", "blosum45", "blosum50", "blosum62", "blosum80", "blosum90"]):
        
        if len(self._mutant_catalog) == 0:
            raise ValueError("Library is empty. Cannot load or generate mutants")
        
        if type in ["protmpnn_decoder"]:
            raise TypeError("Embeddings from ProteinMPNN Decoder need to be loaded, since Generating them takes too long")
        
        sequences = []
        for mutant in self._mutant_catalog:
            mutations = mutant.split(":")

            mutant_sequence = self._wildtype_seq
            for mutation in mutations:
                mutated_pos = int(mutation[1:-1])
                mutant_sequence[mutated_pos-1] = mutation[:-1]
        
        representations = []
        for seq in sequences:
            representations.append(generate_sequence_encodings(type, seq))
        
        return representations


def proper_time(time_in_seconds):
    total_seconds = int(time_in_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    formatted_diff = f"{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_diff


def make_folds(x, y, k_folds=1, shuffle=True):
    """
    Manual KFold splitter. Returns a list of (train_x, train_y, val_x, val_y) tuples.
    - If k_folds=1 → single split, all data in train, empty val
    - If k_folds>1 → k disjoint folds
    - If k_folds='loo' → leave-one-out
    """
    assert isinstance(k_folds, int) or k_folds == "loo", "k_folds must be int > 0 or 'loo'"

    n = len(x)
    indices = list(range(0,n))
    if shuffle:
        random.shuffle(indices)

    folds = []

    if k_folds == 1:
        # all train, no validation
        folds.append([x, y, [], []])

    elif k_folds == "loo":
        for i in range(n):
            val_idx = [indices[i]]
            train_idx = np.delete(indices, i)
            folds.append([x[train_idx], y[train_idx], x[val_idx], y[val_idx]])

    else:  # standard K-fold
        fold_size = math.ceil(n / k_folds)
        for i in range(k_folds):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = copy.copy(indices)
            for idx in val_idx:
                train_idx.remove(idx)
            x_fold_train = [x[idx] for idx in train_idx]
            y_fold_train = [y[idx] for idx in train_idx]
            x_fold_val = [x[idx] for idx in val_idx]
            y_fold_val = [y[idx] for idx in val_idx]
            folds.append([x_fold_train, y_fold_train, x_fold_val, y_fold_val])

    return folds