import src.generate_encodings as ge
import src.prediction_models as pm
import tqdm
import os, sys
import numpy as np
from joblib import parallel_backend
import gc
import time
import random
import warnings

# embedding_type = ["blosum45", "blosum50", "blosum62", "blosum80", "blosum90", "georgiev"]
embedding_type = sys.argv[1]
model_choice = [sys.argv[2]]
# model_choice = ["lightgbm"]
iteration = int(sys.argv[3])
# iteration = 3

data_set = "HIS7_YEAST_Pokusaeva_2019"
# import_data = f"../Data/{data_set}.csv"
# import_data = f"../Data/Protein_Gym_Datasets/{data_set}.csv"
import_data = f"../Data/{data_set}.csv"


# for model in ["rf", "gboost", "xgboost", "xgboost_rf"]:
#     model_choice.append(model)


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


x = []
y = []

with open(import_data, "r") as infile:
    for line in infile.readlines()[1:]:
        line = line[:-1].split(",")
        sequence = line[1]
        emb1 = ge.generate_sequence_encodings(embedding_type, [sequence])[0].tolist()
        if embedding_type == "georgiev":
            for aa in emb1:
                filled_aa = aa.append(float(0))

        ohe = ge.generate_sequence_encodings("one_hot", [sequence])[0].tolist()
        for aa in ohe:
            emb1.append(aa)

        emb1 = np.array(emb1)

        x.append(emb1)
        y.append(line[2])

print(f"successfully created all {embedding_type} + OHE embeddings and loaded them")

k_folds = 5

benchmark_results = dict()

number_examples = 1

results_file = "../../Results/results_HIS7_2_emb.csv"
single_results_file = "../Results/results_HIS_singles_2emb.csv"

for j, m_type in enumerate(model_choice):
    scores = []
    best_iteration = 0
    best_score = 999
    for i in range(number_examples):
        with parallel_backend('threading', n_jobs=18):
            # use random seed
            model = pm.ActivityPredictor(model_type=m_type, data=(x, y), x_column_index=0,
                                         y_column_index=1)
            with HiddenPrints():
                start_time = time.time()
                with HiddenWarnings():
                    model.train(k_folds)
                    val_score = model.get_performance()
                scores.append(val_score)

            with open(single_results_file, "a") as fast_out:
                fast_out.write(
                    f"{embedding_type}+OHE_{m_type}\t{iteration}\t{(round(float(val_score[0]), 3), round(float(val_score[1]), 3))}\n")
            print(
                f"{j}. {embedding_type}+OHE_{m_type}: Iteration {i + 1}/{number_examples}, Duration: {round(time.time() - start_time, 2)}s, Performance: {(round(float(val_score[0]), 3), round(float(val_score[1]), 3))}")

            if val_score[1] < best_score:
                best_score = val_score[1]
                best_iteration = i
                with HiddenPrints():
                    model.save_model(
                        f"../../Results/{data_set}/best_{embedding_type}_OHE_{m_type}_{iteration}/near_ground_truth")

            del model
            gc.collect()

            combination_key = f'{embedding_type}+OHE_{m_type}'
            benchmark_results.update({combination_key: scores})
            with open(results_file, "a") as outfile:
                outfile.write(
                    f"{embedding_type}+OHE_{m_type} \t {[(round(float(a), 3), round(float(b), 3)) for a, b in benchmark_results[combination_key]]} \n")

    print("Done. Models have been tested and the best ones saved.")
