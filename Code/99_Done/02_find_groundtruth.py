import src.generate_encodings as ge
import src.prediction_models as pm
import tqdm
import os, sys
from joblib import parallel_backend
import gc

encoding_choice = [sys.argv[1]]
model_choice = [sys.argv[2]]

# for model in ["rf", "gboost", "xgboost", "xgboost_rf"]:
#     model_choice.append(model)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


import_data = "../../Data/HIS7_YEAST_Pokusaeva_2019.csv"
input_data = []
with open(import_data, "r") as infile:
    for line in infile.readlines():
        input_data.append(line[:-1].split(","))

k_folds = 5

model_data = input_data[:]
to_encode = [line[1] for line in input_data[1:]]
benchmark_results = dict()

number_examples = 10

results_file = "../../Results/results_HIS7.csv"

for y, e_type in enumerate(encoding_choice):
    embeddings = []
    with tqdm.tqdm(total=len(to_encode)) as pbar:
        batch_size = 5000
        for i in range(0, len(to_encode), batch_size):
            embeddings += ge.generate_sequence_encodings(method=e_type, sequences=to_encode[i:i + batch_size])
            pbar.update(batch_size)

    print(f'generating {e_type} encodings for {len(to_encode)} done')
    print(f' {len(encoding_choice) - (1 + y)} encoding types to follow')

    for i, embedding in enumerate(embeddings):
        # index i+1 because of header line, [0] to replace the sequence line
        model_data[i + 1][1] = embedding

    with tqdm.tqdm(total=(len(encoding_choice) * len(model_choice) * number_examples)) as pbar:
        for m_type in model_choice:
            scores = []

            best_iteration = 0
            best_score = 999
            with parallel_backend('threading',n_jobs=8):
                for i in range(number_examples):
                    # use random seed
                    model = pm.ActivityPredictor(model_type=m_type, data=model_data, x_column_index=1,
                                                 y_column_index=2)
                    with HiddenPrints():
                        model.train(k_folds)
                        val_score = model.get_performance()
                        scores.append(val_score)

                        single_results_file = "../../Results/results_HIS7_singles.csv"
                        with open(single_results_file, "a") as fast_out:
                            fast_out.write(
                                f"{e_type}_{m_type} \t {(round(float(val_score[0]), 3), round(float(val_score[1]), 3))} \n")

                        if val_score[1] < best_score:
                            best_score = val_score[1]
                            best_iteration = i
                            model.save_model(f"../../Results/best_{e_type}_{m_type}/near_ground_truth")

                        del model
                        gc.collect()

                    pbar.update()

            combination_key = f'{e_type}_{m_type}'
            benchmark_results.update({combination_key: scores})
            with open(results_file, "a") as outfile:
                outfile.write(
                    f"{e_type}_{m_type} \t {[(round(float(a), 3), round(float(b), 3)) for a, b in benchmark_results[combination_key]]} \n")

    print("Done. Models have been tested and the best ones saved.")
