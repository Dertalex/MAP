import time
import src.generate_encodings as ge
import src.prediction_models as pm
from tqdm import tqdm
import os, sys
from joblib import parallel_backend
import gc
import torch
import random
import warnings


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


torch.cuda.empty_cache()
gc.collect()
# random.seed(0)


pLM = "esmc_300m"
repr_layer = "-1"
model_choice = [sys.argv[1]]
iteration = (sys.argv[2])
model_data = []

"""
Load ESM Representations
Warning: Only use this block or the block below. Not both!
"""

data_set = "HIS7_YEAST_Pokusaeva_2019"
path_to_embeddings = f"../Data/Embeddings/{data_set}/{pLM}/{repr_layer}"
mapping_file = f"../Data/Embeddings/{data_set}/pairings.csv"

with open(mapping_file) as f:
    data_pairs = [line[:-1] for line in f.readlines()[1:]]
    random.shuffle(data_pairs)

with tqdm(total=len(data_pairs)) as pbar:
    for line in data_pairs:
        line = line[:-1].split(',')
        embedding = f"{path_to_embeddings}/{line[0][1:]}"
        representation = torch.load(embedding).to(dtype=torch.float32).cpu().numpy()
        score = float(line[1])
        model_data.append((representation, score))
        pbar.update(1)

print("successfully loaded all embeddings and their scores to cpu")
number_examples = 1

results_file = "../../Results/results_GFP.csv"
k_folds = 5

benchmark_results = dict()

for j, m_type in enumerate(model_choice):
    scores = []
    best_iteration = 0
    best_score = 999
    for i in range(number_examples):
        with parallel_backend('threading', n_jobs=18):
            # use random seed
            model = pm.ActivityPredictor(model_type=m_type, data=model_data, x_column_index=0,
                                         y_column_index=1)
            with HiddenPrints():
                start_time = time.time()
                with HiddenWarnings():
                    model.train(k_folds)
                    val_score = model.get_performance()
                scores.append(val_score)
            single_results_file = "../Results/results_HIS_esmc300_singles.csv"

            with open(single_results_file, "a") as fast_out:
                fast_out.write(
                    f"{pLM}_{m_type}\t{iteration}\t{(round(float(val_score[0]), 3), round(float(val_score[1]), 3))}\n")
            print(
                f"{j}. {pLM}_{m_type}: Iteration {i + 1}/{number_examples}, Duration: {round(time.time() - start_time,2)}s, Performance: {(round(float(val_score[0]), 3), round(float(val_score[1]), 3))}")

            if val_score[1] < best_score:
                best_score = val_score[1]
                best_iteration = i
                with HiddenPrints():
                    model.save_model(f"../../Results/{data_set}/best_{pLM}_{m_type}_{iteration}/near_ground_truth")

            del model
            gc.collect()

        combination_key = f'{pLM}_{m_type}'
        benchmark_results.update({combination_key: scores})
        with open(results_file, "a") as outfile:
            outfile.write(
                f"{pLM}_{m_type} \t {[(round(float(a), 3), round(float(b), 3)) for a, b in benchmark_results[combination_key]]} \n")

    print("Done. Models have been tested and the best ones saved.")
