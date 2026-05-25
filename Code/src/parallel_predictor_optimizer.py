#THIS SCRIPT HAS BEEN MESSED UP TO INVESTIGATE ALL PARAMETERS FOR rf AND ridge SIMULTANEOUSLY. DO NOT USE FOR OTHER PURPOSES!

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


class Parallel_Optimizer:
    _model_type = None
    _large_embedding_params = False
    _cv_folds = 5
    _direction = ["maximize"]
    _n_trials = 50
    _early_stopping = False
    _x_arr = None
    _y_arr = None
    _initial_params = {}
    _is_optimized = False
    _best_trial = None
    _best_params = None
    _db_name = None
    _enable_continue = False
    _show_progress = False
    _show_prints = False
    _reversed_order = False

    def __init__(self,
                 model_type: Literal[
                     "rf", "xgboost", "gxboost_rf", "lightgbm", "svr", "adaboost", "ridge", "lasso", "elastic_net", "fnn"],
                 cv_folds, x_arr: list, y_arr: list,
                 n_trials: int, 
                 initial_params, 
                 db_name: str, 
                 enable_continue: bool = False, 
                 early_stopping: float = False, 
                 ):

        self._model_type = model_type
        self._initial_params = initial_params
        self._cv_folds = cv_folds
        self._n_trials = n_trials
        self._x_arr = x_arr
        self._y_arr = y_arr
        self._early_stopping = early_stopping
        self._enable_continue = enable_continue
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


    def _objective(self, trial: optuna.Trial, params) -> tuple[Any, Any]:

        if self._model_type == "xgboost":
            if not self._large_embedding_params:
                params['subsample'] = trial.suggest_float('subsample', 0.4, 1)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.4, 1)
                params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.01, 0.3)
                params['max_depth'] = trial.suggest_int('max_depth', 1, 12)
                params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.01, 5)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.001, 5.0, log=True)
                params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
            
                if params['learning_rate'] < 0.05:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 500, 2000)
                else:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                    
            else: #Params for huge dimensional data and few datapoints
                params['subsample'] = trial.suggest_float('subsample', 0.4, 1)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.01, 0.3)  # CHANGED: Much lower for high-dim
                params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.01, 0.3)  # ADDED
                params['max_depth'] = trial.suggest_int('max_depth', 3, 8)  # CHANGED: Lower depth for high-dim
                params['min_child_weight'] = trial.suggest_float('min_child_weight', 1, 20)  # CHANGED: Higher to prevent overfitting
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.1, 100.0, log=True)
                params['reg_lambda'] = trial.suggest_float('reg_lambda', 1.0, 100.0, log=True) 
                if params['learning_rate'] < 0.05:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 500, 2000)
                else:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)

        if self._model_type == "lightgbm":
            if not self._large_embedding_params:
                params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 30)
                params['num_leaves'] = trial.suggest_int('num_leaves', 2, 30)
                params['min_data_in_bin'] = trial.suggest_int('min_data_in_bin', 1, 30)
                params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.01, 1)
                params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.001, 10, log=True)
                params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.001, 10, log=True)
                params["learning_rate"] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['max_bin'] = trial.suggest_int('max_bin', 10, 100)
                params["n_estimators"] = trial.suggest_int('n_estimators', 50, 300)
                params["bagging_fraction"] = trial.suggest_float('bagging_fraction', 0.5, 1)

            else: #Params for huge dimensional data and few datapoints
                params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 100)  # CHANGED: Higher minimum
                params['num_leaves'] = trial.suggest_int('num_leaves', 8, 64)  # CHANGED: More leaves for capacity
                params['min_data_in_bin'] = trial.suggest_int('min_data_in_bin', 5, 50)  # CHANGED
                params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.001, 0.1)  # CHANGED: Much lower for high-dim
                params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.1, 100, log=True)  # CHANGED: Higher regularization
                params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.1, 100, log=True)  # CHANGED: Higher regularization
                params["learning_rate"] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['max_bin'] = trial.suggest_int('max_bin', 63, 255)  # CHANGED: Higher for more precision
                params["n_estimators"] = trial.suggest_int('n_estimators', 50, 300)
                params["bagging_fraction"] = trial.suggest_float('bagging_fraction', 0.5, 1)

        if self._model_type == "svr":
            if not self._large_embedding_params:
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
                params['cache_size'] = 2000  # ADDED: Larger cache for performance
                params['C'] = trial.suggest_float('C', 0.01, 1000, log=True)
                
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 7)

                try:
                    if params['kernel'] in ['poly', 'rbf', 'sigmoid']:
                        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                except Exception:
                    pass

            else: #Params for huge dimensional data and few datapoints
                
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf'])  # CHANGED: Removed poly/sigmoid (too slow for high-dim)
                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])
                params['cache_size'] = 2000  # ADDED: Larger cache for performance
                params['C'] = trial.suggest_float('C', 0.001, 100, log=True)  # CHANGED: Lower C for regularization
                
                try:
                    if params['kernel'] == 'rbf':
                        params['gamma'] = trial.suggest_float('gamma', 1e-6, 1e-3, log=True)  # CHANGED: Explicit gamma range for high-dim
                except Exception:
                    pass
                    
        if self._model_type == "rf":
            if not self._large_embedding_params:
                params['n_estimators'] = trial.suggest_int('n_estimators', 100, 300)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 5, 50)
                params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.01, 0.05, 0.1])  # ADDED: Feature subsampling

                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 15)
                params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.3)
                params['max_depth'] = trial.suggest_int('max_depth', 2, 20)  # ADDED: Limit depth
                
            else: #Params for huge dimensional data and few datapoints
                params['n_estimators'] = trial.suggest_int('n_estimators', 100, 300)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 5, 50)  # CHANGED: Higher for high-dim
                params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.01, 0.05, 0.1])  # ADDED: Feature subsampling
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 2, 30)  # CHANGED: Higher minimum
                params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.3)
                params['max_depth'] = trial.suggest_int('max_depth', 5, 20)  # ADDED: Limit depth
                
                
        if self._model_type == "adaboost":
            # no one should ever use adaboost for regression tasks...
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 2.0)


        if self._model_type == "ridge":
            #params should in dimensional space of 1e2 to 1e4
            params['alpha'] = trial.suggest_float('alpha', 0.0001, 1000, log=True)
            params['solver'] = trial.suggest_categorical('solver',
                                                            ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag",
                                                            "saga"])
            params['tol'] = trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            params['max_iter'] = 20000

        if self._model_type == "lasso":
            #params should in dimensional space of 1e2 to 1e4
            params['alpha'] = trial.suggest_float('alpha', 0.001, 1000, log=True)
            params['selection'] = trial.suggest_categorical('selection', ["cyclic", "random"])
            params['tol'] = trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            params['max_iter'] = 20000

        if self._model_type == "elastic_net":
            #params should in dimensional space of 1e2 to 1e4
            params['alpha'] = trial.suggest_float('alpha', 0.001, 1000, log=True)
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            params['tol'] = trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            params['max_iter'] = 25000

        if self._model_type == "fnn":
            #since i have no idea, what i should do here, it does not matter whether to add additional param spaces for huge dim data.
            #If this does not work, the architecture might need to change
            params['n_layers'] = trial.suggest_int("n_hidden_layers", 0, 5)
            params['n_units'] = trial.suggest_categorical("n_units", [128, 256, 512, 1024, 2048])
            params['lr'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            params['dropout'] = trial.suggest_float("dropout", 0.1, 0.7)
            params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        results = self._train_with_params(params)

        trial.set_user_attr("ndcg", round(float(results[0]), 4))
        trial.set_user_attr("spearman", round(float(results[1]), 4))
        trial.set_user_attr("pearson", round(float(results[2]), 4))
        trial.set_user_attr("r2", round(float(results[3]), 4))
        trial.set_user_attr("mse", round(float(results[4]), 4))

        gc.collect()
        
        return round(float(results[1]), 4) #spearman


    def optimize(self, 
                          large_embeddings: bool = False,
                          show_progress: bool = True, 
                          show_prints=False):

        '''
        Start the Optimization. 
        To handle embeddings of proteins represented with over 10k features (i.e. proteinmpnn_decoder or other)
        set large embeddings to True to select an alternative parameter range
        '''
        if  large_embeddings:
            self._large_embedding_params = True
    
        if show_progress:
            self._show_progress = True

        if show_prints:
            self._show_prints = True

        if self._is_optimized:
            warnings.warn(
                "Ideal parameters have already been identified for this configuration. "
                "The Optimizer is not intended to be executed multiple times. Please create a new instance.")
            return

        study = optuna.create_study(study_name=f"MLDE-Model",
                                    storage=f"sqlite:///{self._db_name}.db" if self._enable_continue else None,
                                    load_if_exists=self._enable_continue
                                    )
                                    
        study.optimize(lambda trial: self._objective(trial, self._initial_params), self._n_trials,
                       show_progress_bar=self._show_progress)


        print("=========================== FINAL OPTIMAL PARAMETERS ============================")
        print(f'Highest Achieved Scores: {study.best_trial}')
        print("EVALUATION METRIC: ", "Spearman, (NDCG/Spearman/Pearson/R2/MSE)")
        print(f"BEST TRIAL:, {study.best_trial.number} of {self._n_trials}")
        print(f"BEST SCORE:, {study.best_trial.value}, ({study.best_trial.user_attrs['ndcg'], study.best_trial.user_attrs['spearman'], study.best_trial.user_attrs['pearson'], study.best_trial.user_attrs['r2'], study.best_trial.user_attrs['mse']})")
        print(f'final params: {study.best_trial.params} \n')

        self._is_optimized = True
        self._best_trial = study.best_trial
        self._best_params = study.best_trial.params  
        print('------------------------------------------------')
        

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
