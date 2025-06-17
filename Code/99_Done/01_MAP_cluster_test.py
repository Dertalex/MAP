import generate_encodings as ge
import prediction_models as pm
import tqdm
import os, sys
from joblib import parallel_backend
import torch

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import_data = "../data/NOD.csv"
input_data = []
with open(import_data, "r") as infile:
    for line in infile.readlines():
        input_data.append(line[:-1].split(","))

k_folds = 5
model_data = input_data
to_encode = [line[0] for line in input_data[1:]]
benchmark_results = dict()

e_types = ["esm2_3B"][0:]
models = ["xgboost", "xgboost_rf", "rf", "svr", "adaboost", "gboost"][0:]
number_examples = 50

for e_type in e_types:
    encodings = ge.generate_sequence_encodings(method=e_type, sequences=to_encode)

    with tqdm.tqdm(total=(len(e_types) * len(models) * number_examples)) as pbar:
        for i, encoding in enumerate(encodings):
            # index i+1 because of header line, [0] to replace the sequence line
            model_data[i + 1][0] = encoding
        with parallel_backend('threading', n_jobs=12):
            for m_type in models:
                scores = []
                for i in range(number_examples):
                    # use random seed
                    model = pm.ActivityPredictor(model_type=m_type, data=model_data, x_column_index=0,
                                                 y_column_index=2)
                    with HiddenPrints():
                        model.train(k_folds)
                        scores.append(model.get_performance())
                    pbar.update()
                benchmark_results.update({f'{e_type}_{m_type}': scores})

esm2_results = benchmark_results
out = "results_esm2.csv"
with open(out, "w") as outfile:
    for key in esm2_results.keys():
        outfile.write(f"{key} \t {[(round(float(a), 3), round(float(b), 3)) for a, b in benchmark_results[key]]} \n")
