import random
import src.generate_encodings as ge
import src.prediction_models as pm
import src.predictor_optimizer as pop
import src.mutant_discovery as dis
import numpy as np
import warnings
import os, sys
import torch
from sklearn.metrics import root_mean_squared_error


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


#dataset = "D7PM05_CLYGR_Somermeyer_2022"
dataset = "GRB2_HUMAN_Faure_2021"
dataset_name_shortened = dataset.split("_")[0]
embedding_types = ["one_hot", "blosum45", "blosum50", "blosum62", "blosum80", "blosum90", "georgiev", "esmc_600m"]
models = ["xgboost", "lightgbm"]
cv_folds = 5
esf = 0.1
n_trials = 200
n_jobs = 1


import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
sequences = []
scores = []

for embedding_type in embedding_types[-1:]:
    for model in models:
        if embedding_type not in ["esmc_600m", "esmc_300m"]:
            with open(import_data, "r") as infile:
                for i, line in enumerate(infile.readlines()[1:]):
                    line = line[:-1].split(",")
                    sequence = line[1]
                    label = line[2]
                    sequences.append(sequence)
                    scores.append(label)
                    if i == 0:
                        wildtype_seq = line[5]
            data = [(sequence, ge.generate_sequence_encodings(embedding_type, [sequence])[0], label) for sequence, label in
                    zip(sequences, scores)]
        else: # embedding_type in ["esmc_600m", "esmc_300m"]

            dirs = f"../Data/Embeddings/{dataset}"
            pairings_file = os.path.join(dirs,"pairings.csv")
            repr_layer = 3
            path_to_embeddings = f"../Data/Embeddings/{dataset}/{embedding_type}/{repr_layer}"
            data = []

            with open(pairings_file, "r") as infile:
                lines = infile.readlines()[1:]
            for line in lines:
                line = line[:-1].split(",")
                embedding = os.path.join(path_to_embeddings, line[0])
                data.append(("",torch.load(embedding).to(dtype=torch.float32).cpu(),float(line[1])))


        del sequences, scores

        # three studys for every encoding-model-combination
        n_studies = range(3)
        results = []

        for run in n_studies:
            random.shuffle(data)
            x_arr = [sxy[1] for sxy in data]
            y_arr = [sxy[2] for sxy in data]

            optimizer = pop.Sequential_Optimizer(model_type=model,
                                                 cv_folds=cv_folds,
                                                 x_arr=x_arr,y_arr=y_arr,
                                                 initial_params={},
                                                 trials_per_group=n_trials,
                                                 early_stopping_fraction=esf,
                                                 n_jobs= n_jobs
                                                 )

            optimizer.optimize_stepwise()
            best_trial = optimizer.get_best_trial()
            best_params = optimizer.get_best_params()
            results.append((run,best_trial.values,best_params))

        outpath = "../Models/Hypertuned/sequential/"
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        outfile = f"{dataset_name_shortened}_{model}_{embedding_type}.txt"
        with open(os.path.join(outpath, outfile), "a") as out:
            out.write(f"Study_Details: {model}_{embedding_type}_{dataset}\n")
            out.write("\n")
            out.write("------------------------------------------------\n")
            for run, scores, params in results:
                out.write(f"Study-No: {run}\n")
                out.write("\n")
                out.write(f"Achieved Scores: {scores}\n")
                out.write(f"Identified Params: {params}\n")
                out.write("------------------------------------------------\n")

