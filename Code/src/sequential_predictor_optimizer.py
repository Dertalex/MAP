import gc
import src.prediction_models as pm
import src.metrics as metrics
from typing import Literal, Optional, Any
import optuna
from copy import copy
import warnings
import os, sys
from datetime import datetime
import time

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HiddenWarnings:
    def __enter__(self):
        # Save the current filter settings before changing them
        self._previous_filters = warnings.filters[:]
        # Ignore all warnings
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original warning filter settings
        warnings.filters = self._previous_filters


class Sequential_Optimizer:
    _model_type = None
    _cv_folds = 5
    _direction = ["maximize"]
    _n_trials = 500
    _early_stopping = False
    _x_arr = None
    _y_arr = None
    _initial_params = {}
    _is_optimized = False
    _best_trial = None
    _best_params = None
    _db_name = None
    _show_progress = False
    _show_prints = False
    _reversed_order = False

    def __init__(self,
                 model_type: Literal[
                     "rf", "xgboost", "gxboost_rf", "lightgbm", "linear", "svr", "adaboost", "ridge", "lasso", "elastic_net", "fnn"],
                 cv_folds, x_arr: list, y_arr: list, 
                 initial_params, 
                 trials_per_group: int,
                 db_name: str, 
                 early_stopping: float = False, 
                 reverse_optimization_order: bool = False):

        self._model_type = model_type
        self._initial_params = initial_params
        self._cv_folds = cv_folds
        self._n_trials = trials_per_group
        self._x_arr = x_arr
        self._y_arr = y_arr
        self._early_stopping = early_stopping
        self._reversed_order = reverse_optimization_order
        self._db_name = db_name

    def _train_with_params(self, params: dict = {}) -> (float, float):
        optuna.logging.set_verbosity(optuna.logging.FATAL)

        regressor = pm.ActivityPredictor(model_type=self._model_type, x_arr=self._x_arr, y_arr=self._y_arr,
                                         params=params, early_stopping=self._early_stopping, shuffle_data=False)

        with HiddenWarnings():
            if not self._show_prints:
                with HiddenPrints():
                    regressor.train(k_folds=self._cv_folds)
            else:
                regressor.train(k_folds=self._cv_folds)
            performance = regressor.get_performance()
        return performance

    def _objective(self, trial: optuna.Trial, group, params) -> tuple[Any, Any]:

        if self._model_type == "xgboost":
            # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-"+score.__name__)
            # params["callbacks"] = [pruning_callback]
            if group == 0:
                pass

            if group == 1:
                params['subsample'] = trial.suggest_float('subsample', 0.4, 1)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.4, 1)

            if group == 2:
                params['max_depth'] = trial.suggest_int('max_depth', 1, 12)
                params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.01, 5)

            if group == 3:
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
                if params['learning_rate'] < 0.05:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 500, 2000)
                else:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)

            if group == 4:
                params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.001, 5.0, log=True)
                params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)

        if self._model_type == "lightgbm":

            if group == 0:
                pass

            if group == 1:
                params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 30)
                params['num_leaves'] = trial.suggest_int('num_leaves', 2, 30)

            if group == 2:
                params['min_data_in_bin'] = trial.suggest_int('min_data_in_bin', 1, 30)
                params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.01, 1)

            if group == 3:
                params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.001, 10, log=True)
                params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.001, 10, log=True)

            if group == 4:
                params["learning_rate"] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['max_bin'] = trial.suggest_int('max_bin', 10, 100)

            if group == 5:
                params["n_estimators"] = trial.suggest_int('n_estimators', 50, 300)
                params["bagging_fraction"] = trial.suggest_float('bagging_fraction', 0.5, 1)

        if self._model_type == "svr":
            if group == 0:
                pass
            if group == 1:
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 7) # will be ignored, if kernel not poly

                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
                params['C'] = trial.suggest_float('C', 0.01, 1000, log=True)
                try:
                    if params['kernel'] in ['poly', 'rbf', 'sigmoid']:
                        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                except KeyError: 
                    pass            

            if group == 2:
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 7) # will be ignored, if kernel not poly

                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
                params['C'] = trial.suggest_float('C', 0.01, 1000, log=True)
                try:
                    if params['kernel'] in ['poly', 'rbf', 'sigmoid']:
                        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                except KeyError: 
                    pass            

            if group == 3:
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 7) # will be ignored, if kernel not poly

                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
                params['C'] = trial.suggest_float('C', 0.01, 1000, log=True)
                try:
                    if params['kernel'] in ['poly', 'rbf', 'sigmoid']:
                        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                except KeyError: 
                    pass                            
                
        if self._model_type == "rf":
            if group == 0:
                pass
            if group == 1:
                params['n_estimators'] = trial.suggest_int('n_estimators', 100, 300)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 30)
            if group == 2:
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 15)
                params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.3)

        if self._model_type == "adaboost":
            if group == 0:
                pass
            if group == 1:
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 1000)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 10)

        if self._model_type == "ridge":
            if group == 0:
                pass
            if group == 1:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['solver'] = trial.suggest_categorical('solver',
                                                             ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag",
                                                              "saga"])
                params['tol'] = 0.0001
                params['max_iter'] = 10000
            
            if group == 2:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['solver'] = trial.suggest_categorical('solver',
                                                             ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag",
                                                              "saga"])
                params['tol'] = 0.0001
                params['max_iter'] = 10000

        if self._model_type == "lasso":
            if group == 0:
                pass
            if group == 1:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['selection'] = trial.suggest_categorical('selection', ["cyclic", "random"])
                params['tol'] = 0.0001
                params['max_iter'] = 10000

            if group == 2:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['selection'] = trial.suggest_categorical('selection', ["cyclic", "random"])
                params['tol'] = 0.0001
                params['max_iter'] = 10000

        if self._model_type == "linear":
            if group == 0:
                pass

        if self._model_type == "elastic_net":
            if group == 0:
                pass
            if group == 1:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                params['tol'] = 0.0001
                params['max_iter'] = 15000
                
            if group == 2:
                params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                params['tol'] = 0.0001
                params['max_iter'] = 15000
                

        if self._model_type == "fnn":
            if group == 0:
                pass
            if group == 1:  # architecture + learning rate
                params['n_layers'] = trial.suggest_int("n_hidden_layers", 0, 5)
                params['n_units'] = trial.suggest_categorical("n_units", [128, 256, 512, 1024, 2048])
                params['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                params['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
                
            if group == 2:  # everything again...
                params['n_layers'] = trial.suggest_int("n_hidden_layers", 0, 5)
                params['n_units'] = trial.suggest_categorical("n_units", [128, 256, 512, 1024, 2048])
                params['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                params['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
        
        print(f"Applied Hyperparams: {params}")
                
        results = self._train_with_params(params)

        ndcg = round(float(results[0]), 4)
        spearman = round(float(results[1]), 4)
        pearson = round(float(results[2]), 4)
        r2 = round(float(results[3]), 4)
        mse = round(float(results[4]), 4)

        trial.set_user_attr("ndcg", ndcg)
        trial.set_user_attr("spearman", spearman)
        trial.set_user_attr("pearson", pearson)
        trial.set_user_attr("r2", r2)
        trial.set_user_attr("mse", mse)

        gc.collect()
        return spearman

    def _execute_optimization(self, study_name, group, n_trials, params : dict = {}):
        study = optuna.create_study(study_name=study_name,
                                    directions=self._direction,
                                    storage=None
                                    )
                                    
        study.optimize(lambda trial: self._objective(trial, group, params), n_trials=n_trials,
                       show_progress_bar=self._show_progress)

        if group == 0:
            print("EVALUATION METRIC: ", "Spearman, (NDCG/Spearman/Pearson/R2/MSE)")
            print(
                f"Default SCORE:, {study.trials[0].value}, ({study.trials[0].user_attrs['ndcg'], study.trials[0].user_attrs['spearman'], study.trials[0].user_attrs['pearson'], study.trials[0].user_attrs['r2'], study.trials[0].user_attrs['mse']})")

        else:
            time.sleep(2)
            
            completed_trials = list(t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            failed_trials = list(t for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
            
            if len(completed_trials) ==0:
                return study.trials[0]    
            
        
            print(f'n completed trials: {len(completed_trials)}')
            print(f'n failed trials: {len(failed_trials)}')
            print("STUDY NAME: ", study_name)
            print("EVALUATION METRIC: ", "Spearman, (NDCG/Spearman/Pearson/R2/MSE)")
            print(
                f"BEST SCORE:, {study.best_trial.value}, ({study.best_trial.user_attrs['ndcg'], study.best_trial.user_attrs['spearman'], study.best_trial.user_attrs['pearson'], study.best_trial.user_attrs['r2'], study.best_trial.user_attrs['mse']})")
            print(f"OPTIMAL PARAMS FOR GROUP{group}: ", study.best_trial.params)
            print("BEST TRIAL:", study.best_trial.number)
            print('------------------------------------------------')
                
        return study.best_trial if group != 0 else study.trials[0]

    def optimize_stepwise(self, show_progress: bool = True, show_prints=False):

        if show_progress:
            self._show_progress = True

        if show_prints:
            self._show_prints = True

        if self._is_optimized:
            warnings.warn(
                "Ideal parameters have already been identified for this configuration. "
                "The Optimizer is not intended to be executed multiple times. Please create a new instance.")
            return

        best_group = 0
        final_best_trial = None

        n_groups = {"xgboost": 4,
                    "lightgbm": 5,
                    "rf": 2,
                    "adaboost": 2,
                    "svr": 3,
                    "linear": 0,
                    "ridge": 2,
                    "lasso": 2,
                    "elastic_net": 2,
                    "fnn": 2}

        identified_params = dict()
        optimization_order = range(0, n_groups[self._model_type] + 1) if self._reversed_order is False else range(
            n_groups[self._model_type] + 1, 0, -1)
        for i, group in enumerate(optimization_order):

            if i == 0:
                print(f"=========================== Default Configuration ============================")
                initial_trial = self._execute_optimization("MLDE-Model", group=group, n_trials=1,
                                                           params=self._initial_params)
                final_best_trial = initial_trial
                identified_params = self._initial_params
                print()

            else:
                print(f"============================ Optimizing Group - {group} ============================")
                study_result = self._execute_optimization(study_name=f"MLDE-Model_Parameter-Group {group}", group=group,
                                                          n_trials=self._n_trials, params=copy(identified_params))

                before = copy(final_best_trial.value) if final_best_trial.value is not None else -1e3
                try:
                    if before < study_result.value: #assuming maximization of score (i.e. no (R)MSE)
                        final_best_trial = study_result
                        best_group = group
                        identified_params.update(study_result.params)
                        print(f"SCORE IMPROVED! IDEAL PARAMS UPDATED AS FOLLOWED:\n"
                            f"{identified_params}")
                
                    else:
                        print(f"SCORE DID NOT IMPROVE! PARAMETERS FROM LAST STUDY HAVE BEEN MAINTAINED\n")
                        
                except TypeError:
                    print(f"OPTIMIZATION FOR GROUP {group} FAILED COMPLETELY. PARAMETERS FROM LAST STUDY HAVE BEEN MAINTAINED.\n")
                    


        print("=========================== FINAL OPTIMAL PARAMETERS ============================")
        print(f'Best Study: {best_group}')
        print(f'Highest Achieved Scores: {final_best_trial.values}')
        print(f'final params: {identified_params} \n')

        self._is_optimized = True
        self._best_trial = final_best_trial
        self._best_params = identified_params

        return

    def get_best_trial(self):
        if self._is_optimized:
            return self._best_trial
        else:
            warnings.warn("Optimizer must be executed first.")
            return

    def get_best_params(self):
        if self._is_optimized:
            return self._best_params
        else:
            warnings.warn("Optimizer must be executed first.")
            return
