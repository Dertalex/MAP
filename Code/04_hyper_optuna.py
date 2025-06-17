import random
import src.prediction_models as pm
from typing import Literal, Optional
import src.generate_encodings as ge
import optuna
import warnings

import os, sys

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


def train_with_params(model: Literal["rf", "xgboost", "gxboost_rf", "lightgbm", "svr", "adaboost"], x_arr, y_arr,
                      params: dict,
                      early_stopping: Optional[int] = False, seed: Optional[int] = random.seed()) -> (float, float):
    regressor = pm.ActivityPredictor(model_type=model, x_arr=x_arr, y_arr=y_arr, params=params,
                                     early_stopping=early_stopping, shuffle_data=False,
                                     seed=seed)
    
    with HiddenPrints():
        with HiddenWarnings():
            regressor.train(k_folds=cv_folds)
    performance = regressor.get_performance()
    print('------------------------------------------------')
    print(f"R2: {round(performance[0], 4)}, RMSE: {round(performance[1], 4)}")
    return performance


def objective(trial: optuna.Trial) -> (float, float):
    params = dict()
    seed = 42
    if model == "xgboost":
        params['max_depth'] = trial.suggest_int('max_depth', 3, 30)
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.01, 10)
        params['subsample'] = trial.suggest_float('subsample', 0.2, 1)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.2, 0.9)
        params['eta'] = trial.suggest_float('eta', 0, 0.3)

    if model == "lightgbm":
        params['num_boost_round'] = trial.suggest_int('num_boost_round', 50, 400)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.0, 1.0)
        params['num_leaves'] = trial.suggest_int('num_leaves', 36, 2048)
        params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.0, 1.0)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 30)
        params['max_bin'] = trial.suggest_int('max_bin', 50, 1023)

    if model == "rf":
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 400)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 5)
        params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 1.0)

    if model == "adaboost":
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 1000)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.0, 10000)


    try:
        early_stopping = int(params['num_boost_round'] * 15)
    except KeyError:
        early_stopping = 15

    results = train_with_params(model, x_arr, y_arr, params, early_stopping=early_stopping, seed=seed)
    r2 = round(results[0], 4)
    rmse = round(results[1], 4)

    return r2, rmse





"""Extract Sequence-Embeddings and Scores """
dataset = sys.argv[1]
e_type = sys.argv[2]
model = sys.argv[3]
cv_folds = 5
run_id = int(sys.argv[4]) #number of study
n_trials = int(sys.argv[5])
direction = ["maximize", "minimize"]


data_set_name_shortened = dataset.split("_")[0]
import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"

x_arr = []
y_arr = []
with open(import_data, "r") as infile:
    for line in infile.readlines()[1:]:
        line = line[:-1].split(",")
        sequence = line[1]
        label = line[2]
        x_arr.append(ge.generate_sequence_encodings(e_type, [sequence])[0])
        y_arr.append(label)

studies_name = f"{data_set_name_shortened}_{model}_{e_type}"
overall_best_trials = []

study_name = f"{studies_name}_{run_id}"

study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name)

with HiddenPrints():
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
print('------------------------------------------------')
for b_trial in study.best_trials:
    overall_best_trials.append((b_trial, run_id))
    print(
        f"{study_name}, Best hyperparameters for run {run_id}:, Best Trial: {b_trial.number}, Best Score: {b_trial.values}, "
        f"Best Params{b_trial.params}")

outpath = "../Models/Hypertuned/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
outfile = study_name + ".txt"

with open(os.path.join(outpath, outfile), "w") as out:
    out.write(f"Study-ID: {run_id}_{model}_{e_type}_{dataset}\n")
    out.write("\n")
    for j, b_trial in enumerate(study.best_trials):

        out.write(
            f"Study {j}, Best hyperparameters for run {run_id}:\nBest Trial: {b_trial.number}\nBest Score: {b_trial.values}\n"
            f"Best Params: {b_trial.params} \n")
        if j < len(study.best_trials)-1:
            out.write("---\n")

    out.write("\n")
    for trial in study.trials:
        out.write('------------------------------------------------\n')
        out.write(f"Study {trial.number}\nScores: {trial.values}\n")
        out.write(f"Parameters: {trial.params}\n")
print("Done!")
