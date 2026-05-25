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
import torch
import re

########################################################### Definitions ###########################################################
start_time_stamp = time.time()

debugging = False
demo_case = False
update_params_after_each_cycle = True
skip_initial_hyperparameter_tuning = False
starting_points_determination = "ZS-ΔΔG_guided"

"""defining Parameters for Input_Data"""

# dataset params
dataset = sys.argv[1]
repr_type = sys.argv[2]  # blosum metrics, OHE, Georgiev, ESM
score_column = "Norm_Score_1"  # if true all data ranges from 0 to 1, with the activity threshold differs for every dataset

"""Defining further ModelParameter"""

# model params
model_type = sys.argv[3]  # xgboost, rf, lightgbm, adaboost, svr, linear, ridge, lasso
cv_folds = 5
early_stopping_rounds = 10

# mlde params
r_top = 0.95  # Percentage cutoff of top scoring datapoints, targeted to be identified during the MLDE Cycles
n_starting_points = int(sys.argv[4])  # realistic, up to [100, 200, 500, 1000, 2000, 5000] points to be expected as common practice -> sys arg
n_starting_point_trials = 1000  # amount of attempts to sample proper starting points
r_top_start = 0.15  # Decimal of mutants with the highest ddG to select starting points from 0.1 therefore means top 10% of the mutants with the highest ddG values.
delta_excl = 0.5  # Threshold to exclude the mutants with Zero-Shot score above


# hypertuning params
initial_trials = 50  # to create an initial parameter setup for the model to begin with
target_metric = "spearman"  # "spearman", "ndcg", "pearson", "mse","r2"

# Setup Up Benchmark ID, Timestamp and Output Directory

array_ID = sys.argv[5]
hyperopt_ID = f'{dataset}_{repr_type}_{model_type}_{n_starting_points}'


#check if params already available, abort script in case 
results_path = "../Results/Initial_params"
if os.path.exists(os.path.join(results_path, f"initial_params_{hyperopt_ID}.txt")):
    sys.exit(0)

# Configure logging
if repr_type in ["esmc_600m", "esmc300m", "prot_t5"]:
    log_path = f"initial_params_logs_plm/{array_ID}-{hyperopt_ID}.log"
else:
    log_path = f"initial_params_logs/{array_ID}-{hyperopt_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler(sys.stdout)]
)

# Logging...
logging.info("============================== MLDE Initial Hyperparams Log ==============================")
logging.info(f"model: {model_type}")
logging.info(f"dataset: {dataset}")
logging.info(f"n_starting_points: {n_starting_points}")
logging.info(f"representation: {repr_type}")

#check if params already available, abort script in case 
results_path = "../Results/Initial_params"
if os.path.exists(os.path.join(results_path, f"initial_params_{hyperopt_ID}.txt")):
    logging.info(f"Initial Params for this configuration already exist. Script has been canceled.")
    sys.exit(0)

# prepare Dataset and Embeddings
import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
data = []  # id, sequence, embedding (x), score (y),zs_score ddG (z)
headers = []

if repr_type in ["esmc_600m", "esmc300m"]:
    # check recommended esmc_layer to be applied for the scenario
    scenario_id = f"{dataset}_{model_type}_{n_starting_points}"
    layer_recommendations_file = "layer_recommendations.txt"
    recommended_layer = 0
    with open(layer_recommendations_file, "r") as lrf:
        recommendation_lines = lrf.readlines()
    for line in recommendation_lines:
        if line.startswith(scenario_id):
            recommended_layer = int(line.split(":")[1].split("_")[2].strip())
            break

with open(import_data, "r") as infile:
    lines = infile.readlines()
for i, line in enumerate(lines[0:]):
    line = line[:-1].split(",")
    if i == 0:
        headers = line
        continue

    id = str(line[0])
    sequence = str(line[1])
    if repr_type not in ["esmc_600m", "esmc300m", "prot_t5"]: # plm embeddings should be loaded due to time expenses
        embedding = ge.generate_sequence_encodings(repr_type, [sequence])[0]
    
    else:
        # load precomputed embeddings

        if repr_type in ["esmc_600m", "esmc300m"]: 
            emb_path = f"../Data/Embeddings/{dataset}/{repr_type}/{recommended_layer}" 
        
        else: 
            emb_path = f"../Data/Embeddings/{dataset}/{repr_type}"
            
        embedding = torch.load(os.path.join(emb_path, f"{id}.pt"),
                                map_location=torch.device('cpu'))

    label = round(float(line[headers.index(score_column)]), 3)
    zs_score = round(float(line[4]), 3)
    data.append((id, sequence, embedding, label, zs_score))

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
stretch = 1
while n_starting_points > len(library)*r_top_start*stretch:
    stretch += 0.2

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

gc.collect()  # clear memory

################################## Benchmark ###################################

finished = False
hyperopt_timestamp = time.time()
cause_of_termination = "Benchmark did not achieve the target score during the maximum allowed number of cycles."

logging.info("============================= Determining starting points =============================\n")

mlde_params = {}

mlde_optimizer = pop.Sequential_Optimizer(model_type=model_type, cv_folds=cv_folds, x_arr=x_mlde, y_arr=y_mlde,
                                            initial_params=copy(mlde_params),
                                            trials_per_group=initial_trials,
                                            db_name=f"{hyperopt_ID}_0",
                                            early_stopping=early_stopping_rounds,
                                            )

mlde_optimizer.optimize_stepwise(show_progress=False, show_prints=False)  # the optimizer will compare current hyperparams with with "altered params" and will only overrite them in case of improvement
best_trial = mlde_optimizer.get_best_trial()

if mlde_params != mlde_optimizer.get_best_params():
    mlde_params = mlde_optimizer.get_best_params()  # might even overwrite with the same params
    params_updated = True
else:
    params_updated = False

finish_timestamp = time.time()
logging.info(f"Hyperparameters updated." if params_updated else " Current Hyperparameters maintained.")
logging.info(f"Duration of Cycle's Hyperparameter Tuning: {proper_time(finish_timestamp - hyperopt_timestamp)}\n")

out_path = "../Results/Initial_params/"
out_file = os.path.join(out_path, f"initial_params_{hyperopt_ID}.txt")
with open(out_file, "w") as f:
    f.write(f'params: {mlde_params}')
