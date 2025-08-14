import logging
from math import sqrt
import random
import src.generate_encodings as ge
import src.prediction_models as pm
import src.predictor_optimizer as pop
from src.metrics import *
from copy import copy
import warnings
from src.utils import HiddenPrints, HiddenWarnings, proper_time
import os, sys
from tqdm import tqdm
from datetime import datetime
import numpy as np
import gc
import time
import re

########################################################### Definitions ###########################################################
start_time_stamp = time.time()

debugging = False
demo_case = False
update_params_after_each_cycle = True
skip_initial_hyperparameter_tuning = False
starting_points_determination = "ZS-ΔΔG_guided"
benchmark_run = int(sys.argv[1])  # run number of the benchmark, to be used for the file name, 20 in total

"""defining Parameters for Input_Data"""

# dataset params
dataset = sys.argv[2]
repr_type = sys.argv[3]  # blosum metrics, OHE, Georgiev, ESM
score_column = "Norm_Score_1"  # if true all data ranges from 0 to 1, with the activity threshold differs for every dataset

"""Defining further ModelParameter"""

# model params
model_type = sys.argv[4]  # xgboost, rf, lightgbm, adaboost, svr, linear, ridge, lasso
cv_folds = 5
early_stopping_fraction = 0.01

# mlde params
r_top = 0.97  # Percentage cutoff of top scoring datapoints, targeted to be identified during the MLDE Cycles
n_starting_points = int(sys.argv[5])  # realistic, up to [100, 200, 500, 1000, 2000, 5000] points to be expected as common practice -> sys arg
n_starting_point_trials = 1000  # amount of attempts to sample proper starting points
n_gain = int(sys.argv[6])  # realistic range of obtained samples per iteration: [10,20,50,100] -> sys arg
n_attempts = 50  # Number of training attempts per cycle to build a better performing model than for the previous cylcle.
r_top_start = 0.15  # Decimal of mutants with the highest ddG to select starting points from 0.1 therefore means top 10% of the mutants with the highest ddG values.
n_cycles = 40  # Since one cycle is expected to take at least 1 month (lab validation), I do not expect the whole project to take for more than 5 years.
delta_excl = 0.5  # Threshold to exclude the mutants with Zero-Shot score above
n_cancel = 5  # number of rounds to cancel the Benchmark after if no higher scoring mutant has been found in the last n_cancel rounds.

# hypertuning params
initial_trials = 50  # to create an initial parameter setup for the model to begin with
n_trials = 25  # trials per group to optimize the parameters for the mlde model - after each cycle.
target_metric = "spearman"  # "spearman", "ndcg", "pearson", "mse","r2"

# Setup Up Benchmark ID, Timestamp and Output Directory

array_ID = sys.argv[7]


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
benchmark_ID = f"{array_ID}-{dataset}_{repr_type}_{model_type}_nGain{n_gain}_nStart{n_starting_points}_ddG_top_start{round(r_top_start, 2)}_nCycles{n_cycles}_normedScores_run{benchmark_run}"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{benchmark_ID}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Logging...
logging.info("============================== MLDE Benchmark Log ==============================")
logging.info(f"Benchmark ID: {benchmark_ID}")
logging.info(f"Timestamp: {timestamp}")
logging.info("")

# prepare Dataset and Embeddings
if repr_type not in ["esmc_600m", "esmc300m"]:
    import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
    data = []  # id, sequence, embedding (x), score (y),zs_score ddG (z)
    headers = []

    with open(import_data, "r") as infile:
        for i, line in enumerate(infile.readlines()[0:]):
            line = line[:-1].split(",")
            if i == 0:
                headers = line
                continue

            id = line[0]
            sequence = line[1]
            embedding = ge.generate_sequence_encodings(repr_type, [sequence])[0]
            label = round(float(line[headers.index(score_column)]), 3)
            zs_score = round(float(line[4]), 3)
            data.append((id, sequence, embedding, label, zs_score))

else:  # load referring ESM Embeddings
    logging.info("Loading ESM Embeddings not implemented yet.")

# Determining Parameters for MLDE Benchmark
wild_type = data[0][1]
mutations = data[0][0].split(":")
for mut in mutations:
    index = int(mut[1:-1]) - 1
    wild_type = wild_type[:index] + mut[0] + wild_type[index + 1:]
max_score = max([isxyz[3] for isxyz in data])
max_starting_score = round(max([isxyz[3] for isxyz in data]) * r_top_start, 3)
target_score = round(max_score * r_top, 3)

# Define the library of potential mutants by filtering out those with a ΔΔG above the threshold
library = [isxyz for isxyz in data if isxyz[4] <= delta_excl]
logging.info("################### Dataset-Parameters ###################")
logging.info(f"Applied Data-Set: {dataset}")
logging.info(f"Number of datapoints: {len(data)}")
logging.info(f"Wildtype Sequence: {wild_type}")
logging.info(f"Library-Size (=Number of mutants within threshold): {len(library)}")
logging.info(f"ΔΔG-Threshold for Exclusion: >{delta_excl} kcal/mol")
logging.info(f"maximum score: {max([isxyz[3] for isxyz in data])}")
logging.info(f"minimum score: {min([isxyz[3] for isxyz in data])}\n")

logging.info("#################### Model-Parameters ####################")
logging.info(f"Protein Representation Type: {repr_type}")
logging.info(f"Algorithm: {model_type}")
logging.info(f"Cross Validation Folds: {cv_folds}")
logging.info(f"Early Stopping Fraction (if applicable): {early_stopping_fraction}\n")

logging.info("##################### MLDE-Parameters ####################")
logging.info(f"Number of Cycles: {n_cycles}")
logging.info(f"Number of Starting Points: {n_starting_points}")
logging.info(f"Number of Gain Points per Cycle: {n_gain}")
stretch = 1
while n_starting_points > len(library)*r_top_start*stretch:
    stretch += 0.2
logging.info(f"Effective Top ΔΔG-Percentage for Starting Points Selection: {int(r_top_start*stretch*100)}%")
logging.info(f"Target Score to achieve ({int(r_top * 100)}% of Top-Score): {target_score}")
logging.info(f"Number of Training Attempts per Cycle: {n_attempts}\n")

logging.info("############ Hyper-Parameter-Tuning-Parameters ###########")
logging.info(f"n Trials for initial Hyper-Parameter Tuning: {initial_trials}")
logging.info(f"n Trials for Hyper-Parameter Tuning after each Cycle: {n_trials}")
logging.info(f"n Parameter to optimize for: {target_metric}\n")

########################################################### Selecting Datapoints for Benchmark ###########################################################
determined_starting_points = False
trial_counter = 0

# stretch factor to increase the range of the top starting points, if not enough are available

while not determined_starting_points and trial_counter < n_starting_point_trials:
    try:
        upper_bound = int(stretch *r_top_start * len(library))+1
        mlde_datapoints = random.sample(sorted(library, key=lambda isxyz: isxyz[4])[0:upper_bound], int(n_starting_points))
        determined_starting_points = True
    except ValueError as e:
        pass
    finally:
        trial_counter += 1

if not determined_starting_points:
    logging.error("[ERROR] Amount of available, allowed Amount of Active or Inactive Starting Points does not meet defined criteria!")
    logging.error(f"The script will be stopped, since no proper starting points could be determined within {n_starting_point_trials} trials .")
    sys.exit(1)

sequences_mlde = [isxyz[1] for isxyz in mlde_datapoints]
x_mlde = [isxyz[2] for isxyz in mlde_datapoints]
y_mlde = [isxyz[3] for isxyz in mlde_datapoints]

starting_points = copy(mlde_datapoints)

remaining_data = [isxyz for isxyz in library if isxyz not in mlde_datapoints]
random.shuffle(remaining_data)
target_score = r_top * max_score

initialization_time_stamp = time.time()
logging.info(f"Declared MLDE Starting-Points and remaining dataset within {trial_counter} trials")
if stretch > 1:
    logging.info(f"The Top ΔΔG-Percentage for Starting Points Selection needed to be increased effectively to: {round(1 - r_top_start*stretch, 2)}%")
else:
    logging.info(f"The Top ΔΔG-Percentage for Starting Points Selection of {round(1 - r_top_start*stretch, 2)} % was sufficient")

logging.info(f"Duration of Initialization: {proper_time(initialization_time_stamp - start_time_stamp)}\n")

######################## Initial Hyper-Parameter-Tuning #######################
"""Create a suitable hyperparameter-set for the initial MLDE-Model"""
logging.info("======================= Initial Hyperparameter-Definition ======================\n")
if skip_initial_hyperparameter_tuning:
    logging.info(" Skipping initial Hyper-Parameter Tuning for MLDE Model.")
    logging.info(f" Duration of Initial Hyper-Parameter Tuning: 00:00:00")

    mlde_params = {}
else:
    mlde_optimizer = pop.Sequential_Optimizer(model_type=model_type, cv_folds=cv_folds, x_arr=x_mlde, y_arr=y_mlde,
                                              initial_params={},
                                              trials_per_group=initial_trials,
                                              early_stopping_fraction=early_stopping_fraction,
                                              db_name=f"{benchmark_ID}_0")

    mlde_optimizer.optimize_stepwise()
    best_trial = mlde_optimizer.get_best_trial()
    mlde_params = mlde_optimizer.get_best_params()

    try:
        os.remove(f"optuna_study-{benchmark_ID}_0.db")
    except Exception as e:
        logging.warning(f"[WARNING] Could not remove Optuna Database for initial Hyper-Parameter Tuning: {e}")
        pass

    initial_tuning_time = time.time() - initialization_time_stamp
    logging.info(f"Initial Hyper-Parameter Tuning for {model_type} model completed.")
    logging.info(f"Duration of Initial Hyper-Parameter Tuning: {proper_time(initial_tuning_time)}\n")

gc.collect()  # clear memory

################################## Benchmark ###################################

finished = False
mlde_start_time_stamp = time.time()
cause_of_termination = "Benchmark did not achieve the target score during the maximum allowed number of cycles."

logging.info("============================= MLDE-Benchmark-Start =============================\n")
logging.info(f"Starting the MLDE-Benchmark with {n_cycles} cycles, starting at {n_starting_points} sequences.")
logging.info(f"Discovery of {n_gain} per Cycle to identify the highest scoring sequences >= {target_score} (i.e. {int(r_top * 100)}%) of max from list.")
logging.info(f"Initial Hyper-Parameters for {model_type} model:\n{mlde_params}\n")

scored_mutants = [["mutant", "predicted_score", "true_score"]]  # save each iterations highest achieved sequence score and the average over all sequences and standard deviation
train_performances = ["Spearman,NDCG,Pearson,R2,MSE"]  # Performance for each cycle's trained model on the training data
val_performances = ["Spearman,NDCG,Pearson,R2,MSE"]  # Performance for each cycle's trained model on the validation data
test_performances = ["Spearman,NDCG,Pearson,R2,MSE"]  # Performance for each cycle's tested model on the remaining data

previous_Spearman = float(-1)
current_Spearman = float(-1)
best_cycle_Spearman = float(-1)

highest_achieved_score = -999
cycles_without_improvement = 0  # Counter for cycles without improvement

for i in range(1, n_cycles + 1):
    cycle_start_time_stamp = time.time()

    logging.info(f"######################## Starting Cycle {i}/{n_cycles} #########################\n")

    best_cycle_model = None
    j = 0
    logging.info(f"starting the training attempts")
    while j < n_attempts:
        mlde_model = pm.ActivityPredictor(model_type=model_type,
                                        x_arr=x_mlde,
                                        y_arr=y_mlde,
                                        shuffle_data=True,
                                        early_stopping=10,
                                        params=mlde_params)
        with HiddenPrints():
            with HiddenWarnings():
                mlde_model.train(k_folds=cv_folds)
                current_Spearman = round(mlde_model.get_performance()[0], 3)
                current_Pearson = mlde_model.get_performance()[1]

        if current_Pearson is float('nan') or current_Spearman is float('nan'):
            continue # skip this iteration if Pearson or Spearman is NaN, which can happen with some models

        j += 1

        if current_Spearman > best_cycle_Spearman or best_cycle_model is None:
            best_cycle_Spearman = copy(current_Spearman)
            best_cycle_model = copy(mlde_model)

    # obtain Trainingsperformance
    y_train_pred = best_cycle_model.predict(best_cycle_model.get_data(prepared=True)["x_train"])
    y_train = best_cycle_model.get_data(prepared=True)["y_train"]

    train_NDCG = float(round(ndcg_score([y for y in y_train_pred], [y for y in y_train]), 3))
    train_Spearman = float(round(spearman_correlation([y for y in y_train_pred], [y for y in y_train]), 3))
    train_Pearson = float(round(pearson_correlation([y for y in y_train_pred], [y for y in y_train]), 3))
    train_R2 = float(round(r2_score([y for y in y_train_pred], [y for y in y_train]), 3))
    train_MSE = float(round(mse([y for y in y_train_pred], [y for y in y_train]), 3))

    val_NDCG = float(round(best_cycle_model.get_performance()[0], 3))
    val_Spearman = float(round(best_cycle_model.get_performance()[1], 3))
    val_Pearson = float(round(best_cycle_model.get_performance()[2], 3))
    val_R2 = float(round(best_cycle_model.get_performance()[3], 3))
    val_MSE = float(round(best_cycle_model.get_performance()[4], 3))

    cycle_finish_time = proper_time(time.time() - cycle_start_time_stamp)

    logging.info(f"Best attempt's Val-Performance after {j} training attempts: ")
    logging.info(f"Spearman: {val_Spearman}, (NDCG {val_NDCG}, Pearson: {val_Pearson}, R2: {val_R2}, MSE: {val_MSE})")
    logging.info(f"Training Performance for the best Attempt: ")
    logging.info(f"Spearman: {train_Spearman}, (NDCG {train_NDCG}, Pearson: {train_Pearson}, R2: {train_R2}, MSE: {train_MSE})")

    # Predict on the remaining data
    list_predictions = []
    logging.info(f"Predicting on the remaining {len(library)} datapoints...")
    for isxy in remaining_data:
        list_predictions.append(best_cycle_model.predict([isxy[2]])[0])

    test_NDCG = float(round(ndcg_score([y for y in list_predictions], [isxy[3] for isxy in remaining_data]), 3))
    test_Spearman = float(round(spearman_correlation([y for y in list_predictions], [isxy[3] for isxy in remaining_data]), 3))
    test_Pearson = float(round(pearson_correlation([y for y in list_predictions], [isxy[3] for isxy in remaining_data]), 3))
    test_R2 = float(round(r2_score([y for y in list_predictions], [isxy[3] for isxy in remaining_data]), 3))
    test_MSE = float(round(mse([y for y in list_predictions], [isxy[3] for isxy in remaining_data]), 3))

    logging.info(f"Interference-Performance on all remaining datapoints (incl. Out of Distribution Prediction):\n"
                 f"Spearman: {test_Spearman}, (NDCG {test_NDCG}, Pearson: {test_Pearson}, R2: {test_R2}, MSE: {test_MSE})\n")
    logging.info("\n")

    top_predictions = sorted([(isxy, y_head) for isxy, y_head in zip(remaining_data, list_predictions)],
                             key=lambda tuple: tuple[1], reverse=True)[:n_gain]

    top_predictions = sorted(top_predictions, key=lambda tuple: tuple[0][3], reverse=True)
    logging.info(f'Top {50} identified Samples:\n')
    iterations_scored_mutants = []
    for target in top_predictions:
        logging.info(f'{target[0][0]}, Predicted: {round(float(target[1]), 3)}, True: {target[0][3]}')
        if target[0][3] >= target_score:
            finished = True
        iterations_scored_mutants.append(f"{target[0][0]}, {round(float(target[1]), 3)}, {target[0][3]}")
    scored_mutants.append(iterations_scored_mutants)
    logging.info("\n")

    train_performances.append(f'{train_NDCG, train_Spearman, train_Pearson, train_R2, train_MSE}')
    val_performances.append(f'{val_NDCG, val_Spearman, val_Pearson, val_R2, val_MSE}')
    test_performances.append(f'{test_NDCG, test_Spearman, test_Pearson, test_R2, test_MSE}')

    if finished:
        cause_of_termination = "target score achieved successfully."
        cycle_train_and_interference_time_stamp = time.time()
        logging.info(f" Duration of Cycle's Training and Interference: {proper_time(cycle_train_and_interference_time_stamp - cycle_start_time_stamp)}")
        logging.info(f" Duration of Cycle {i} in total: {proper_time(time.time() - cycle_start_time_stamp)}\n")

        break

    current_highest_score = top_predictions[0][0][3]
    if current_highest_score > highest_achieved_score:
        highest_achieved_score = current_highest_score
        cycles_without_improvement = 0
    else:
        cycles_without_improvement += 1

    if cycles_without_improvement >= n_cancel:
        cycle_train_and_interference_time_stamp = time.time()
        finished = False
        logging.info(f" Duration of Cycle's Training and Interference: {proper_time(cycle_train_and_interference_time_stamp - cycle_start_time_stamp)}")
        logging.info(f" Duration of Cycle {i} in total: {proper_time(time.time() - cycle_start_time_stamp)}\n")
        cause_of_termination = "stuck in local optimum. Plateau reached."
        break

    # update mlde trainingpoints-range
    for target in top_predictions:
        x_mlde.append(target[0][2])
        y_mlde.append(target[0][3])

    # shuffle the mlde_data (in fact not necessary, but better save than sorry)
    mlde_data = [(x, y) for x, y in zip(x_mlde, y_mlde)]
    random.shuffle(mlde_data)
    x_mlde = [xy[0] for xy in mlde_data]
    y_mlde = [xy[1] for xy in mlde_data]

    # update remaining data
    remaining_data = [isxy for isxy in remaining_data if isxy not in [isxy_yhead[0] for isxy_yhead in top_predictions]]
    random.shuffle(remaining_data)

    highest_score = max([target[0][3] for target in top_predictions])
    mean_score = round((sum([target[0][3] for target in top_predictions]) / n_gain), 3)
    standard_dev = round(
        sqrt(sum([(y - mean_score) ** 2 for y in [target[0][3] for target in top_predictions]]) / n_gain), 3)

    # prepare next cycle:
    # ->update last iterations previous_Spearman
    previous_Spearman = copy(best_cycle_Spearman)

    cycle_train_and_interference_time_stamp = time.time()
    cycle_train_and_interference_time = cycle_train_and_interference_time_stamp - cycle_start_time_stamp

    params_updated = False
    if update_params_after_each_cycle:
        mlde_optimizer = pop.Sequential_Optimizer(model_type=model_type, cv_folds=cv_folds, x_arr=x_mlde, y_arr=y_mlde,
                                                  initial_params=copy(mlde_params),
                                                  trials_per_group=int(n_trials),
                                                  early_stopping_fraction=early_stopping_fraction,
                                                  db_name=f"{benchmark_ID}_{i}")

        with HiddenPrints():
            mlde_optimizer.optimize_stepwise()  # the optimizer will compare current hyperparams with with "altered params" and will only overrite them in case of improvement
            best_trial = mlde_optimizer.get_best_trial()

        if mlde_params != mlde_optimizer.get_best_params():
            mlde_params = mlde_optimizer.get_best_params()  # might even overwrite with the same params
            params_updated = True
        else:
            params_updated = False

        try:
            os.remove(f"optuna_study-{benchmark_ID}_{i}.db")
        except Exception as e:
            pass

    cycle_finish_time_stamp = time.time()

    logging.info(f" Hyperparameters updated." if params_updated else " Current Hyperparameters maintained.")
    logging.info(f" Duration of Cycle's Training and Interference: {proper_time(cycle_train_and_interference_time)}")
    logging.info(f" Duration of Cycle's Hyperparameter Tuning: {proper_time(cycle_finish_time_stamp - cycle_train_and_interference_time_stamp)}")
    logging.info(f" Duration of Cycle {i} in total: {proper_time(cycle_finish_time_stamp - cycle_start_time_stamp)}\n")

mlde_end_time_stamp = time.time()

logging.info(f"MLDE-Performance-Ranking finished after {i}/{n_cycles} cycles {'successfully' if finished else 'without success'}.")
logging.info(f" Duration of total MLDE-Benchmark: {proper_time(mlde_end_time_stamp - mlde_start_time_stamp)}\n")

####################### Documenting the Results #######################

# Setting up output directories
out = f'../Results/MLDE_Benchmark/{benchmark_ID}'
if not os.path.exists(out):
    os.makedirs(out)

# document starting_points
with open(f"{out}/01_Starting_Points.txt", "w") as f:
    f.write("mutant,score,zs_score\n")
    for isxyz in starting_points:
        f.write(f"{isxyz[0]},{isxyz[3]},{isxyz[4]}\n")

# document results
files = ["03_Training_Performances.txt", "04_Validation_Performances.txt", "05_Test_Performances.txt", "06_Scored_Mutants.txt"]

for i, data in enumerate([train_performances, val_performances, test_performances, scored_mutants]):
    with open(f"{out}/{files[i]}", "w") as f:
        f.write(f"MLDE-Benchmark Performance Data {files[i][:-4]} for {benchmark_ID} - {timestamp}\n")
        for entry in data:
            f.write(f"{entry}\n")

logging.info(f"Results documented under {out}")
logging.info(f"Benchmark_Result: {cause_of_termination}")
logging.info(f"Duration of total Script Execution: {proper_time(time.time() - start_time_stamp)}")