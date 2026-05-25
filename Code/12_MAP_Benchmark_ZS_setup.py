import logging
import random
import src.generate_encodings as ge
import src.prediction_models as pm
import src.sequential_predictor_optimizer as spop
from src.metrics import *
from copy import copy
from src.utils import *
import os, sys
from datetime import datetime
import gc
import time
import ast
import warnings
import torch
import shutil
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

##################################################### Definitions ######################################################
start_time_stamp = time.time()

debugging = False
demo_case = False
skip_initiation = False
update_params_before_each_cycle = True
load_from_checkpoint = True
starting_points_determination = "ZS-ΔΔG_guided"
benchmark_run = int(sys.argv[1])  # run number of the benchmark, to be used for the file name, 10 in total

"""defining Parameters for Input_Data"""

# dataset params
dataset = sys.argv[2]
repr_type = sys.argv[3]  # blosum metrics, OHE, Georgiev, ESM
if repr_type in ["prost_t5", "mpnn_probs", "esmc_300m", "esmc_600m"]:
    repr_class = "plm" # precomputed, load from disk (the term plm (protein language model) might be actually incorrect here, but who cares)
else:
    repr_class = "otf" #on the fly
    
score_column = "Norm_Score_1"  # if true all data ranges from 0 to 1, with the activity threshold differs for every dataset

"""Defining further ModelParameter"""
# model params
model_type = sys.argv[4]  # xgboost, rf, lightgbm, adaboost, svr, linear, ridge, lasso
cv_folds = 5
early_stopping = 10

# mlde params
r_top = 0.95  # Percentage cutoff of top scoring datapoints, targeted to be identified during the MLDE Cycles
n_starting_points = int(sys.argv[
                            5])  # realistic, up to [100, 200, 500, 1000, 3000] points to be expected as common practice
n_starting_point_trials = 1000  # amount of attempts to sample proper starting points
n_gain = int(sys.argv[6])  # realistic range of obtained samples per iteration: [10,20,50,100]
n_attempts = 20  # Number of training attempts per cycle to train a suitable model. The best attempt will be used for interference
r_top_start = 0.15  # Decimal of mutants with the highest ddG to select starting points from 0.1 therefore means top 10% of the mutants with the highest ddG values.
n_cycles = 40  # Since one cycle is expected to take at least 1 month (lab validation), I do not expect the whole project to take for more than 5 years.
delta_excl = 0.5  # Threshold to exclude the mutants with Zero-Shot score above
n_cancel = 10  # number of rounds to cancel the Benchmark after if no higher scoring mutant has been found in the last n_cancel rounds.

# hypertuning params
n_trials = 25  # trials per group to optimize the parameters for the mlde model - after each cycle.
target_metric = "spearman"  # "spearman", "ndcg", "pearson", "mse","r2"

# Setup Up Benchmark ID, Timestamp and Output Directory

array_ID = sys.argv[7]
results_folder = sys.argv[8]
log_dir=sys.argv[9]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
benchmark_ID = f"{dataset}_{repr_type}_{model_type}_nGain{n_gain}_nStart{n_starting_points}_ddG_top_start{round(r_top_start, 2)}_nCycles{n_cycles}_normedScores_run{benchmark_run}"
destination_dir = os.path.join(results_folder, f"fin_{benchmark_ID}")

initial_params_ID = f'{dataset}_{repr_type}_{model_type}_{n_starting_points}'

# cancel run if already finished
if os.path.exists(destination_dir):
    sys.exit(0) #clean exit. Script already once run completely.
    
    # Configure logging
log_path = f"{log_dir}/{array_ID}_{benchmark_ID}.log"
if os.path.exists(log_path):
    for i in range(1, 1000):
        log_path = f"{log_dir}/{array_ID}_{benchmark_ID}_v{i}.log"
        if not os.path.exists(log_path):
            break

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler(sys.stdout)]
)

# Setting up output directories
out_dir = os.path.join(results_folder,benchmark_ID)
if not os.path.exists(os.path.join(out_dir, "checkpoints")):
    os.makedirs(os.path.join(out_dir, "checkpoints"))

# Logging...
logging.info("============================== MLDE Benchmark Log ==============================")
logging.info(f"Benchmark ID: {benchmark_ID}")
logging.info(f"Current_slurmjob_array: {array_ID}")
logging.info(f"Preparing Protein Representations and Mutant Library...")

# Determine Location of Precomputed Embeddings (if applicable)
if repr_class == "plm":
    if repr_type not in ["esmc_600m", "esmc300m"]:  # currently "prot_t5", "prost_t5", "protmpnn_decoder", "protmpnn"
        emb_location = f"../Data/Embeddings/{dataset}/{repr_type}"
    else: # repr_type in ["esmc_600m", "esmc300m"]:
    # check recommended esmc_layer to be applied for the scenario
        scenario_id = f"{dataset}_{model_type}_{n_starting_points}"
        layer_recommendations_file = "layer_recommendations.txt"
        recommended_layer = 0
        with open(layer_recommendations_file, "r") as lrf:
            recommendation_lines = lrf.readlines()
            for line in recommendation_lines:
                if line.startswith(scenario_id):
                    recommended_layer = int(line.split(":")[1].split("_")[2].strip())
                    repr_type = "_".join((line.split(":")[1]).strip().split("_")[:-1]) #update repr_type to esmc_300m if necessary
                    logging.info(f"Switching representation type to: {repr_type} for benchmark ID due to recommended calculations")
                    break
        emb_location = f"../Data/Embeddings/{dataset}/{repr_type}/{recommended_layer}" 

# prepare Dataset and Embeddings
import_data = f"../Data/Protein_Gym_Datasets/{dataset}.csv"
data = [] # id, sequence, embedding (x), score (y),zs_score ddG (z)
headers = []
          
with open(import_data, "r") as infile:
    lines = infile.readlines()
for i, line in enumerate(lines[0:]):
    line = line[:-1].split(",")
    if i == 0:
        headers = line
        continue

    id = str(line[0])
    sequence = str(line[1])
    label = round(float(line[headers.index(score_column)]), 3)
    zs_score = round(float(line[4]), 3)
    data.append(Protein(id, 
                        sequence, 
                        label, 
                        zs_score))

# Determining Parameters for MLDE Benchmark
wild_type_seq = data[0].determine_wt()  # determine wild-type sequence from first mutant in catalog
max_score = max([mutant._score for mutant in data])
target_score = round(max_score * r_top, 3)

# Define the library of potential mutants by filtering out those with a ΔΔG above the threshold
library = [mutant for mutant in data if mutant._zs_score <= delta_excl]
logging.info(f"Library prepared")

############################################ Load Checkpoint (if available) ############################################

load_from_checkpoint = False
starting_points = []
mlde_params = {}
scored_mutants = {}
cycles_without_improvement = 0
last_cycle = 0

try:
    # Locate previous run directory
    outdirs = os.listdir(results_folder)
    load_dir = next((os.path.join(results_folder, f)
                     for f in outdirs if benchmark_ID == f), None)

    if load_dir is None:
        raise FileNotFoundError("No previous checkpoint directory found")

    # Load starting points
    starting_points_file = os.path.join(load_dir, "starting_points.txt")
    with open(starting_points_file, "r") as infile:
        line = infile.readline()
        if not line.startswith("starting_points:"):
            raise ValueError("Invalid starting points format")
        starting_points = [protein_id.strip() for protein_id in line[16:].split(",")]

    # Load checkpoint data
    checkpoint_dir = os.path.join(load_dir, "checkpoints")
    checkpoint_files = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x[6:-15].strip()))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No checkpoint files found")

    with open(os.path.join(checkpoint_dir, checkpoint_files[-1]), "r") as infile:
        lines = infile.readlines()

    # Parse Checkpoint Content

    last_cycle = int(lines[0].split(":")[-1].strip())
    mlde_params = dict(ast.literal_eval(lines[1][13:].strip()))

    for i, line in enumerate(lines[2:]):
        scored_mutants[i + 1] = [m.strip() for m in line[len(f"cycle {i + 1} scored_mutants:"):].strip().split(",")]

    # calculate cycles without improvement
    top_mutants = []
    idx_library = [mutant._id for mutant in library]
    for cycle_mutants in scored_mutants.values():
        top_mutants.append(max([library[idx_library.index(mutant)]._score for mutant in cycle_mutants]))
    best_cycle = top_mutants.index(max(top_mutants)) + 1
    cycles_without_improvement = last_cycle - best_cycle
    
    for score in top_mutants:
        if score >= target_score:
            logging.info(f"Target score already achieved in previous run. Exiting benchmark.")
            logging.info(f"Resultsfolder renamed accordingly to fin_{benchmark_ID}")
            
            os.rename(out_dir, os.path.join(results_folder,f"fin_{benchmark_ID}"))
            try:
                shutil.copy(log_path, os.path.join(results_folder,log_path.split("/")[-1]))
            except Exception as e:
                logging.info(f"Copying log file to output directory failed: {e}")
                         
            sys.exit(0)           
            
    load_from_checkpoint = True
    logging.info(f"Checkpoint loaded successfully.\n")

except Exception as e:
    load_from_checkpoint = False

############################################## Log Initialization (so far) #############################################

if load_from_checkpoint:
    logging.info(f"Continuing previous run starting from checkpoint at cycle {last_cycle + 1}/{n_cycles}.")
if not load_from_checkpoint:
    logging.info("################### Dataset-Parameters ###################")
    logging.info(f"Applied Data-Set: {dataset}")
    logging.info(f"Number of datapoints: {len(data)}")
    logging.info(f"Wildtype Sequence: {wild_type_seq}")
    logging.info(f"Library-Size (=Number of mutants within threshold): {len(library)}")
    logging.info(f"ΔΔG-Threshold for Exclusion: >{delta_excl} kcal/mol")
    logging.info(f"maximum score: {max([mutant._score for mutant in data])}")
    logging.info(f"minimum score: {min([mutant._score for mutant in data])}\n")

    logging.info("#################### Model-Parameters ####################")
    logging.info(f"Protein Representation Type: {repr_type}")
    logging.info(f"Algorithm: {model_type}")
    logging.info(f"Cross Validation Folds: {cv_folds}")
    logging.info(f"Early Stopping Rounds (if applicable): {early_stopping}\n")

    logging.info("##################### MLDE-Parameters ####################")
    logging.info(f"Number of Cycles: {n_cycles}")
    logging.info(f"Number of Starting Points: {n_starting_points}")
    logging.info(f"Number of Gain Points per Cycle: {n_gain}")
    stretch = 1
    while n_starting_points > len(library) * r_top_start * stretch:
        stretch += 0.2
    logging.info(f"Effective Top ΔΔG-Percentage for Starting Points Selection: {int(r_top_start * stretch * 100)}%")
    logging.info(f"Target Score to achieve ({int(r_top * 100)}% of Top-Score): {target_score}")
    logging.info(f"Number of Training Attempts per Cycle: {n_attempts}\n")

    logging.info("############ Hyper-Parameter-Tuning-Parameters ###########")
    logging.info(f"n Trials for initial Hyper-Parameter Tuning: 50")
    logging.info(f"n Trials for Hyper-Parameter Tuning after each Cycle: {n_trials}")
    logging.info(f"n Parameter to optimize for: {target_metric}\n")

########################################## Selecting Datapoints for Benchmark ##########################################
initialization_time_stamp = time.time()

if load_from_checkpoint:  # load from checkpoint:
    scored_mutants_arr = np.ravel(np.array(list(scored_mutants.values())))
    mlde_datapoints = [mutant for mutant in library if mutant._id in starting_points or mutant._id in scored_mutants_arr]
    remaining_data = [mutant for mutant in library if mutant not in mlde_datapoints]

    del scored_mutants_arr
    logging.info(f"MLDE-Datapoints applied from Checkpoint")

if not load_from_checkpoint:
    determined_starting_points = False
    trial_counter = 0

    # stretch factor to increase the range of the top starting points, if not enough are available

    while not determined_starting_points and trial_counter < n_starting_point_trials:
        try:
            upper_bound = int(stretch * r_top_start * len(library)) + 1
            
            mlde_datapoints = random.sample(sorted([m for m in library if m._score < target_score], 
                                                   key=lambda mutant: mutant._zs_score)[0:upper_bound],
                                            int(n_starting_points))
            determined_starting_points = True
        except ValueError as e:
            pass
        finally:
            trial_counter += 1

    if not determined_starting_points:
        logging.error(
            "[ERROR] Amount of available, allowed Amount of Active or Inactive Starting Points does not meet defined criteria!")
        logging.error(
            f"The script will be stopped, since no proper starting points could be determined within {n_starting_point_trials} trials .")
        sys.exit(1)

    with open(os.path.join(out_dir, "starting_points.txt"), "w") as f:
        f.write(f"starting_points: {', '.join([mutant._id for mutant in mlde_datapoints])}\n")

    remaining_data = [mutant for mutant in library if mutant not in mlde_datapoints]
    target_score = r_top * max_score

    logging.info(f"Declared MLDE Starting-Points and remaining dataset within {trial_counter} trials")
    if stretch > 1:
        logging.info(
            f"The Top ΔΔG-Percentage for Starting Points Selection needed to be increased effectively to: {round(1 - r_top_start * stretch, 2)}%")
    else:
        logging.info(
            f"The Top ΔΔG-Percentage for Starting Points Selection of {round(1 - r_top_start * stretch, 2)} % was sufficient")

logging.info(f"Duration of Initialization: {proper_time(initialization_time_stamp - start_time_stamp)}\n")
        

#################################################### Run Benchmark #####################################################
finished = False
mlde_start_time_stamp = time.time()
cause_of_termination = "Tbd"

start_cycle = 1 if not load_from_checkpoint else last_cycle + 1
if start_cycle >= n_cycles + 1:
    cause_of_termination = "Benchmark did not achieve the target score during the maximum allowed number of cycles."
    finished = True

scored_mutants = dict() if not load_from_checkpoint else scored_mutants  # save each iterations highest achieved sequence score and the average over all sequences and standard deviation
current_Spearman = float(-1)
best_cycle_Spearman = float(-1)
highest_achieved_score = -999

cycles_without_improvement = 0 if not load_from_checkpoint else cycles_without_improvement  # Counter for cycles without improvement
if cycles_without_improvement >= n_cancel:
    cause_of_termination = "stuck in local optimum. Plateau reached."
    finished = True

logging.info("============================= MLDE-Benchmark-Start =============================\n")

logging.info(
    f"Discovery of {n_gain} per Cycle to identify the highest scoring sequences >= {target_score} (i.e. {int(r_top * 100)}%) of max from list.")
logging.info(
    f"Starting the MLDE-Benchmark at cycle {start_cycle}/{n_cycles} cycles, with {len(mlde_datapoints)} known protein variants.\n")

for cycle in range(start_cycle, n_cycles + 1):

    if finished:
        break

    cyclestart_timestamp = time.time()
    logging.info(f"############################ Starting Cycle {cycle}/{n_cycles} #############################\n")

    #load embeddings for current mlde datapoints
    
    computation_device = 'cpu'
    if repr_class == "plm":
        x_mlde = [torch.load(os.path.join(emb_location, f"{mutant._id}.pt"), map_location=torch.device(computation_device)) for mutant in mlde_datapoints]        
    
    #generate embeddings on the fly
    else: #repr_class == "otf" therefore generate on the fly
        x_mlde = ge.generate_sequence_encodings(repr_type, [protein._seq for protein in mlde_datapoints])
    y_mlde = [mutant._score for mutant in mlde_datapoints]
        
    if cycle == 1:

        #Load Hyperparameters at the first cycle
        if model_type != "linear":
            params_location = "../Results/Initial_params"
            params_file = os.path.join(params_location, f"initial_params_{initial_params_ID}.txt")
            with open(params_file, "r") as f:
                content = f.read().strip()
                if content.startswith("params:"):
                    content = content[len("params:"):].strip()
                else:
                    raise ValueError('Expected a params file starting with "params:"')
                mlde_params.update(ast.literal_eval(content))

            logging.info(f"Hyperparameters loaded from params file: {params_file}")
        hyperopt_timestamp = time.time()

    # Start the Hyperparameter-Tuning.
    elif update_params_before_each_cycle:

        # load last iterations hyper-parameters

        logging.info(f"Starting Hyperparametertuning")
        """Update Parameters Before Each Cycle"""

        params_updated = False
        tuning_complete = False
        cancelled_tunings = 0
        max_retries = 1
        cought_exception = None


        
        while not tuning_complete and cancelled_tunings < max_retries:
            try:

                mlde_optimizer = spop.Sequential_Optimizer(model_type=model_type, cv_folds=cv_folds,
                                                          x_arr=x_mlde,
                                                          y_arr=y_mlde,
                                                          initial_params=mlde_params,
                                                          trials_per_group=n_trials,
                                                          early_stopping=early_stopping,
                                                          db_name=f"{benchmark_ID}_{cycle}"
                                                          )

                with HiddenPrints():
                    with HiddenWarnings():
                        mlde_optimizer.optimize_stepwise(
                            show_progress=False)  # the optimizer will compare current hyperparams with with "altered params" and will only overrite them in case of improvement

                tuning_complete = True

            except Exception as e:
                cancelled_tunings = cancelled_tunings + 1
                logging.warning(f'Hyperparameter-Tuning attempt No {cancelled_tunings + 1} failed due to:')
                logging.warning(e)
                cought_exception = e

        if tuning_complete:
            best_trial = mlde_optimizer.get_best_trial()

            if mlde_params != mlde_optimizer.get_best_params():
                mlde_params = mlde_optimizer.get_best_params()  # might even overwrite with the same params
                params_updated = True
            else:
                params_updated = False

            logging.info(f"Hyperparameters updated." if params_updated else " Current Hyperparameters maintained.")

        if not tuning_complete:
            logging.warning(f"Hyperparameters-Tuning failed with repeatedly Exception: {cought_exception}")
            
            #Terminate the whole Benchmark if hyperparameter tuning fails repeatedly, since ensemble appearently might not be suitable for this scenario
            finished = True
            cause_of_termination = "Ensemble is not suitable for this scenario"
            break  # exit the whole cycle if hyperparameter tuning fails repeatedly
            
        
        hyperopt_timestamp = time.time()
        logging.info(
            f"Duration of Cycle's Hyperparameter Tuning: {proper_time(hyperopt_timestamp - cyclestart_timestamp)}")

    """Starting the Training"""
    logging.info("-------------------------------------------------------------------------\n")
    logging.info(f"starting the training attempts")

    best_cycle_model = None
    b = 0
    exception_counts = 0
    cought_exception = None
    while b < n_attempts:
        
        b += 1
        try:
            mlde_model = pm.ActivityPredictor(model_type=model_type,
                                              x_arr=x_mlde,
                                              y_arr=y_mlde,
                                              shuffle_data=True,
                                              early_stopping=early_stopping,
                                              params=mlde_params)
            with HiddenPrints():
                with HiddenWarnings():
                    mlde_model.train(k_folds=cv_folds)
                    current_Spearman = round(mlde_model.get_performance()[0], 3)
                    current_Pearson = mlde_model.get_performance()[1]

            if current_Pearson is float('nan') or current_Spearman is float('nan'):                
                exception_counts += 1
                continue  # skip this attempt if performance metrics are nan
            
            if current_Spearman > best_cycle_Spearman or best_cycle_model is None:
                best_cycle_Spearman = copy.copy(current_Spearman)
                best_cycle_model = copy.copy(mlde_model)

        except Exception as e:
            exception_counts += 1
            cought_exception = e
            pass

    if exception_counts == n_attempts:
        logging.warning(f"All training attempts failed during Cycle {cycle}.")
        if cought_exception is not None:
            logging.warning(f'Last cought Exception: {cought_exception}')
            
        #Terminate the whole Benchmark if model training fails repeatedly, since ensemble appearently might not be suitable for this scenario
        finished = True
        cause_of_termination = "Ensemble is not suitable for this scenario"

    # obtain Trainingsperformance
    if finished:
        break
    
    y_train_pred = best_cycle_model.predict(best_cycle_model.get_data()["x_train"])
    y_train = best_cycle_model.get_data()["y_train"]

    train_NDCG = "NA"
    train_Spearman = float(round(spearman_correlation([y for y in y_train_pred], [y for y in y_train]), 3))
    train_Pearson = float(round(pearson_correlation([y for y in y_train_pred], [y for y in y_train]), 3))
    train_R2 = float(round(r2_score([y for y in y_train_pred], [y for y in y_train]), 3))
    train_MSE = float(round(mse([y for y in y_train_pred], [y for y in y_train]), 3))

    val_NDCG = "NA"
    val_Spearman = float(round(best_cycle_model.get_performance()[1], 3))
    val_Pearson = float(round(best_cycle_model.get_performance()[2], 3))
    val_R2 = float(round(best_cycle_model.get_performance()[3], 3))
    val_MSE = float(round(best_cycle_model.get_performance()[4], 3))

    logging.info(f"Best attempt's Val-Performance after {b} training attempts: ")
    logging.info(f"Spearman: {val_Spearman}, (Pearson: {val_Pearson}, NDCG: {val_NDCG}, R2: {val_R2}, MSE: {val_MSE})")
    logging.info(f"Training Performance for the best Attempt: ")
    logging.info(
        f"Spearman: {train_Spearman}, (Pearson: {train_Pearson}, NDCG: {train_NDCG}, R2: {train_R2}, MSE: {train_MSE})")

    training_timestamp = time.time()
    logging.info(f"Duration of Cycle's Training: {proper_time(training_timestamp - hyperopt_timestamp)}")
    logging.info("-------------------------------------------------------------------------\n")

    # Predict on the remaining data in batches
    logging.info(f"Predicting on the remaining {len(remaining_data)} datapoints...")
    max_retries = 5
    batch_size = len(mlde_datapoints) # if memory was sufficient during training, it should be sufficient here as well

    
    with HiddenPrints():
        with HiddenWarnings():
            list_predictions = []
            start_index = 0
            last_exception = None
            for i in range(max_retries): # max tries for whole list prediction to catch potential memory errors
            
                try:
                    for batch in range(start_index, len(remaining_data), batch_size):
                        if repr_class == "plm":
                            batch_embeddings = [torch.load(os.path.join(emb_location, f"{mutant._id}.pt"), map_location=torch.device(computation_device)) for mutant in remaining_data[batch:batch_size + batch]]

                        else:
                            batch_embeddings = ge.generate_sequence_encodings(repr_type, [protein._seq for protein in remaining_data[batch:batch_size + batch]])

                        batch_predictions = best_cycle_model.predict(batch_embeddings)
                        list_predictions.extend(batch_predictions)
                        start_index = start_index + batch_size
                    
                    break  # exit the retry loop if successful
                                
                except Exception as e:
                    logging.warning(f'Error occured during batched List Prediction: {e}')
                    logging.warning(f'Retries left: {max_retries - i}')
                    batch_size = int(batch_size // 2)  # reduce batch size and retry
                    last_exception = e
                    
            if len(list_predictions) != len(remaining_data):
                raise last_exception("List Prediction Failed.")        
                    
    test_NDCG = "NA"
    test_Spearman = float(
        round(spearman_correlation([y for y in list_predictions], [mutant._score for mutant in remaining_data]), 3))
    test_Pearson = float(
        round(pearson_correlation([y for y in list_predictions], [mutant._score for mutant in remaining_data]), 3))
    test_R2 = float(round(r2_score([y for y in list_predictions], [mutant._score for mutant in remaining_data]), 3))
    test_MSE = float(round(mse([y for y in list_predictions], [mutant._score for mutant in remaining_data]), 3))

    logging.info(f"Interference-Performance on all remaining datapoints (incl. Out of Distribution Prediction):\n"
                 f"Spearman: {test_Spearman}, (Pearson: {test_Pearson}, NDCG: {test_NDCG}, R2: {test_R2}, MSE: {test_MSE})\n")

    # Select the highest predicted mutants as "top predictions" to reveal their true scores and add them to the known mlde_datapoints
    top_predictions = sorted([(mutant, y_head) for mutant, y_head in zip(remaining_data, list_predictions)],
                             key=lambda tuple: tuple[1], reverse=True)[:n_gain]

    
    top_predictions = sorted(top_predictions, key=lambda tuple: tuple[0]._score, reverse=True)
    logging.info(f'Top {50} identified Samples: Predicted, True\n')
    iterations_scored_mutants = []

    for i, (mutant, prediction) in enumerate(top_predictions):
        logging.info(
            f'{mutant._id}, Predicted: {round(float(prediction), 3)}, True: {mutant._score}{"\n" if i == n_gain - 1 else ""}')
        if mutant._score >= target_score:
            finished = True
        iterations_scored_mutants.append(f"{mutant._id},{round(float(prediction), 3)},{mutant._score}")

    scored_mutants.update({cycle: iterations_scored_mutants})
    train_performances = [train_Spearman, train_Pearson, train_NDCG, train_R2, train_MSE]
    val_performances = [val_Spearman, val_Pearson, val_NDCG, val_R2, val_MSE]
    test_performances = [test_Spearman, test_Pearson, test_NDCG, test_R2, test_MSE]

    highest_score = max([mutant._score for mutant, prediction in top_predictions])


    interference_timestamp = time.time()
    logging.info(f"Duration of Cycle's Interference: {proper_time(interference_timestamp - training_timestamp)}")
    logging.info("-------------------------------------------------------------------------\n")

    if finished:
        cause_of_termination = "target score achieved successfully."
        logging.info(f"Duration of Cycle {cycle} in total: {proper_time(time.time() - cyclestart_timestamp)}\n")

    if not finished:  # prepare next cycle:
        current_highest_score = top_predictions[0][0]._score

        if current_highest_score > highest_achieved_score:
            highest_achieved_score = current_highest_score
            cycles_without_improvement = 0
        else:
            cycles_without_improvement += 1

        if cycles_without_improvement >= n_cancel:
            cycle_train_and_interference_time_stamp = time.time()
            finished = False
            logging.info(f"Duration of Cycle {cycle} in total: {proper_time(time.time() - cyclestart_timestamp)}\n")
            cause_of_termination = "stuck in local optimum. Plateau reached."
            break

        # update mlde trainingpoints-range
        for mutant, prediction in top_predictions:
            mlde_datapoints.append(mutant)
            remaining_data.remove(mutant)
        
        random.shuffle(mlde_datapoints)

        cycle_train_and_interference_time_stamp = time.time()
        cycle_train_and_interference_time = cycle_train_and_interference_time_stamp - cyclestart_timestamp

    ##################################### Save Benchmark Results and Checkpoints #######################################

    # Save Hyperparams and scored mutants

    checkpoint_file = f'cycle_{cycle}_checkpoint.txt'
    with open(os.path.join(f"{out_dir}/checkpoints/", checkpoint_file), "w") as f:
        f.write(f"last_cycle: {cycle}\n")
        f.write(f"hyper_params: {mlde_params}\n")
        for c in range(1, cycle + 1):
            f.write(
                f"cycle {c} scored_mutants: {', '.join([mutant.split(',')[0] for mutant in scored_mutants[c]])}\n")

    logging.info(
        f"Scored mutants and hyperparameters for cylce {cycle}/{n_cycles} has been saved at {out_dir}/checkpoints/")

    # Save Model-Performances
    performance_file = "performance.csv"
    if not os.path.exists(os.path.join(f"{out_dir}/{performance_file}")):
        with open(os.path.join(f"{out_dir}/{performance_file}"), "w") as f:
            f.write(f"#Cycle/Training/Validation/Interference")
            f.write(
                f"#Cycle, train_Spearman, train_Pearson, train_NDCG,train_R2,train_MSE, test_Spearman, test_Pearson, test_NDCG,test_R2,test_MSE, interfer_Spearman, interfer_Pearson, Interfer_NDCG,Interfer_R2,Interfer_MSE\n")

    with open(os.path.join(f"{out_dir}/{performance_file}"), "a") as f:
        f.write(
            f"{cycle},{",".join(map(str, train_performances))}, {",".join(map(str, val_performances))}, {",".join(map(str, test_performances))}\n")

    logging.info(f"Performances for Training, Validation and Interference has been saved at {out_dir}\n")

    cyclefinish_timestamp = time.time()
    logging.info(f"Duration of Cycle {cycle} in total: {proper_time(cyclefinish_timestamp - cyclestart_timestamp)}\n")
    try:
        shutil.copy(log_path, os.path.join(out_dir, log_path.split("/")[-1]))
    except Exception as e:
        logging.info(f"Copying log file to output directory failed: {e}")


logging.info(f"Benchmark_Result: {cause_of_termination}\n")
logging.info("=================================== Finished ===================================\n")

os.rename(out_dir, destination_dir)
try:
    shutil.copy(log_path, os.path.join(destination_dir, log_path.split("/")[-1]))
except Exception as e:
    logging.info(f"Copying log file to output directory failed: {e}")