import gc
import src.prediction_models as pm
from typing import Literal, Optional, Any
import optuna
from copy import copy
import warnings
import os, sys
from datetime import datetime

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
    _direction = ["maximize", "minimize"]
    _n_trials = 500
    _n_jobs = -1
    _early_stopping_fraction = False
    _x_arr = None
    _y_arr = None
    _initial_params = {}
    _is_optimized = False
    _best_trial = None
    _best_params = None
    _reversed_order = False

    def __init__(self,
                 model_type: Literal["rf", "xgboost", "gxboost_rf", "lightgbm", "svr", "adaboost", "ridge", "lasso"],
                 cv_folds, x_arr: list, y_arr: list, initial_params, trials_per_group: int,
                 early_stopping_fraction: float, reverse_optimization_order: bool = False, n_jobs: int = -1):

        self._model_type = model_type
        self._initial_params = initial_params
        self._cv_folds = cv_folds
        self._n_trials = trials_per_group
        self._x_arr = x_arr
        self._y_arr = y_arr
        self._early_stopping_fraction = early_stopping_fraction
        self._reversed_order = reverse_optimization_order
        self._njobs = n_jobs

    def _train_with_params(self, params: dict = {}) -> (float, float):
        optuna.logging.set_verbosity(optuna.logging.FATAL)
        if self._early_stopping_fraction:
            if "num_boost_round" in params.keys():
                early_stopping = int(params['num_boost_round'])
            elif "n_estimators" in params.keys():
                early_stopping = int(params["n_estimators"])
            else:
                early_stopping = int(100 * self._early_stopping_fraction)
        else:
            early_stopping = False

        regressor = pm.ActivityPredictor(model_type=self._model_type, x_arr=self._x_arr, y_arr=self._y_arr,
                                         params=params, early_stopping=early_stopping, shuffle_data=False)

        with HiddenWarnings():
            with HiddenPrints():
                # print(performance)
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
                params['subsample'] = trial.suggest_float('subsample', 0.02, 1)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.2, 0.9)

            if group == 2:
                params['max_depth'] = trial.suggest_int('max_depth', 1, 100)
                params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.01, 10)

            if group == 3:
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)

            if group == 4:
                params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 5.0)
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
                params['lambda_l1'] = trial.suggest_float('lambda_l1', 0, 10)
                params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.001, 10, log=True)

            if group == 4:
                params["learning_rate"] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['max_bin'] = trial.suggest_int('max_bin', 4, 300)

            if group == 5:
                params["n_estimators"] = trial.suggest_int('n_estimators', 50, 300)
                params["bagging_fraction"] = trial.suggest_float('bagging_fraction', 0.5, 1)
        
        if self._model_type == "svr":
            if group == 0:
                pass
            if group == 1:
                params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                params['degree'] = trial.suggest_int('degree', 2, 6)
            if group == 2:
                params['C'] = trial.suggest_float('C', 0.01, 1000, log=True)
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            if group == 3:
                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1, log=True)
                params['shrinking'] = trial.suggest_categorical('shrinking', [True, False])


        if self._model_type == "rf":

            if group == 0:
                pass
            if group == 1:
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 30)
            if group == 2:
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 15)
                params['min_weight_fraction_leaf'] = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.9)
            if group == 3:
                params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 1.0)

        if self._model_type == "adaboost":
            if group == 0:
                pass
            if group == 1:
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 1000)
            if group == 2:
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 10)

        if self._model_type == "ridge":
            if group == 0:
                pass
            if group == 1:
                params['alpha'] = trial.suggest_int('alpha', 1, 1000000)
                # params['solver'] = trial.suggest_categorical('solver',
                #                                              ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag",
                #                                               "saga", "lbfgs"])
            if group == 2:
                params['max_iter'] = trial.suggest_int('max_iter', 100, 50000)
                params['tol'] = trial.suggest_float('tol', 0.000001, 0.01)

        if self._model_type == "lasso":
            if group == 0:
                pass
            if group == 1:
                params['alpha'] = trial.suggest_int('alpha', 1, 1000000)
                params['selection'] = trial.suggest_categorical('selection', ["cyclic", "random"])


            if group == 2:
                params['tol'] = trial.suggest_float('tol', 0.000001, 0.01)
                params['max_iter'] = trial.suggest_int('max_iter', 100, 50000)


        results = self._train_with_params(params)
        r2 = round(float(results[0]), 4)
        rmse = round(float(results[1]), 4)

        gc.collect()
        return r2, rmse

    def _execute_optimization(self, study_name, group, n_trials, params=dict(), n_jobs: int = -1):
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        study = optuna.create_study(study_name=study_name, directions=self._direction, storage=f"sqlite:///optuna_study-{timestamp}.db")
        study.optimize(lambda trial: self._objective(trial, group, params), n_trials=n_trials, n_jobs=n_jobs,
                       show_progress_bar=True)
        studys_best_trial = sorted(study.best_trials, key=lambda t: t.values[0])[0]

        if group == 0:
            print("EVALUATION METRIC: ", "R2/RMSE")
            print(f"Default SCORE: {studys_best_trial.values}")

        else:
            print("STUDY NAME: ", study_name)
            print("EVALUATION METRIC: ", "R2/RMSE")
            print(f"BEST SCORE:, {studys_best_trial.values}")
            print(f"OPTIMAL PARAMS FOR GROUP{group}: ", studys_best_trial.params)
            print("BEST TRIAL:", studys_best_trial.number)
            print('------------------------------------------------')

        return studys_best_trial

    def optimize_stepwise(self):

        if self._is_optimized:
            warnings.warn(
                "Ideal parameters have already been identified for this configuration. "
                "The Optimizer is not intended to be executed multiple times. Please create a new instance.")
            return

        best_group = 0
        final_best_trial = None

        n_groups = {"xgboost": 4,
                    "lightgbm": 5,
                    "rf": 3,
                    "adaboost": 2,
                    "ridge": 2,
                    "lasso": 2}

        identified_params = dict()
        optimization_order = range(0, n_groups[self._model_type] + 1) if self._reversed_order is False else range(
            n_groups[self._model_type] + 1, 0, -1)
        for i, group in enumerate(optimization_order):

            if i == 0:
                print(f"=========================== Default Configuration ============================")
                initial_trial = self._execute_optimization("MLDE-Model", group=group, n_trials=1,
                                                           params=self._initial_params, n_jobs=1)
                final_best_trial = initial_trial
                identified_params = self._initial_params
                print()

            else:
                print(f"============================ Optimizing Group - {group} ============================")
                study_result = self._execute_optimization(study_name=f"MLDE-Model_Parameter-Group {group}", group=group,
                                                          n_trials=self._n_trials, params=copy(identified_params),
                                                          n_jobs=self._njobs)

                if final_best_trial.values[0] < study_result.values[0]:
                    final_best_trial = study_result
                    best_group = group
                    identified_params.update(study_result.params)
                    print(f"SCORE IMPROVED! IDEAL PARAMS UPDATED AS FOLLOWED: ,\n"
                          f"{identified_params}")
                else:
                    print(f"SCORE DID NOT IMPROVE! PARAMETERS FROM LAST STUDY HAVE BEEN MAINTAINED")
                    print()

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
