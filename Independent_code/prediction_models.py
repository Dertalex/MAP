from typing import Literal
import pandas as pd
import random
import numpy as np
import warnings

import xgboost
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBRFRegressor

import torch


class ActivityPredictor:
    _model_type = None
    _model = None
    _split = (80, 10, 10)
    _data = None
    _x_column = None
    _y_column = None
    _train_data = None
    _test_data = None
    _val_data = None
    _performance = None
    _roc_auc_score = None
    _is_trained = False
    _seed = None

    def __init__(self, model_type: Literal["rf", "svm", "ada_boost"], data, x_column_index: int, y_column_index: int,
                 split=(80, 10, 10), seed=None):
        self._model_type = model_type
        self._split = split
        self._data = data
        self._x_column = x_column_index
        self._y_column = y_column_index
        self.seed: None
        self._define_model()

    def split_data(self):
        # check if data has headers

        has_header = False if self._data[0][self._y_column].isnumeric() else True
        data = self._data[1:] if has_header else self._data

        if self._seed:
            random.seed(self._seed)
        random.shuffle(data)

        train_data = []
        test_data = []
        val_data = []

        train_size = int(len(data) * self._split[0] / sum(self._split))
        test_size = int(len(data) * self._split[1] / sum(self._split))

        train_data = data[:train_size]
        test_data = data[train_size:train_size + test_size]
        val_data = data[train_size + test_size:]

        return train_data, test_data, val_data

    def _define_model(self):
        models = {"svr": SVR(), "rf": RandomForestRegressor(), "adaboost": AdaBoostRegressor(),
                  "gboost": GradientBoostingRegressor(), "xgboost": XGBRegressor(),
                  "xgboost_rf": XGBRFRegressor()}
        self._model = models[self._model_type]

    def train(self, k_folds):

        self._train_data, self._test_data, self._val_data = self.split_data()

        if self._model_type in ["svr", "rf", "adaboost", "gboost", "xgboost", "xgboost_rf"]:
            # train the model like a scikit-learn model

            # defining the inputs(x) and their label(y)
            x_train = [encoding[self._x_column] for encoding in self._train_data]
            x_val = [encoding[self._x_column] for encoding in self._val_data]
            x_test = [encoding[self._x_column] for encoding in self._test_data]

            y_train = [float(label[self._y_column]) for label in self._train_data]
            y_val = [float(label[self._y_column]) for label in self._val_data]
            y_test = [float(label[self._y_column]) for label in self._test_data]

            # retrieve encoding type from tensor dimensions
            """To Fix: how to multidimensional tensors? (3D and higher)"""

            # converting the datasets in torch.Tensor-Format to numpy arrays
            # flatten the tensors either by view (pytorch-tensor) or reshape (numpy.ndarrays)

            if isinstance(x_train[0], torch.Tensor):
                x_train = [tensor.detach().cpu().numpy() for tensor in x_train]
                x_val = [tensor.detach().cpu().numpy() for tensor in x_val]
                x_test = [tensor.detach().cpu().numpy() for tensor in x_test]

            if len(x_train[0].shape) >= 2:
                x_train = [x.reshape(-1) for x in x_train]
                x_val = [x.reshape(-1) for x in x_val]
                x_test = [x.reshape(-1) for x in x_test]

                """To Fix: other tensor input data types"""

            # create k-fold splits of dataset

            if not k_folds == "loo" and not k_folds.is_integer():
                raise ValueError("k_folds must be an integer or 'loo' for leave-one-out cross-validation")

            if k_folds != 0 or k_folds == "loo":

                x_train = np.concatenate([x_train, x_test])
                y_train = np.concatenate([y_train, y_test])

                model_ensemble = []
                ensemble_scores = []

                if k_folds != "loo":
                    folds = KFold(n_splits=k_folds, shuffle=False, random_state=self._seed)
                else:
                    folds = KFold(n_splits=len(x_train), shuffle=False, random_state=self._seed)

                for i, x_split in enumerate(folds.split(x_train)):
                    train_index, test_index = x_split
                    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
                    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

                    # initiate multiple model instances for ensemble learning
                    self._define_model()
                    self._model.fit(x_train_fold, y_train_fold)
                    model_ensemble.append(self._model)

                self._model = model_ensemble

            else:  # k_folds == 0
                self._define_model()
                # normalized_arr = preprocessing.normalize(x_train, y_train, x_test)
                self._model.fit(x_train, y_train)

                # determine the performance of the model

            self._is_trained = True
            self._performance = self.score(x_val, y_val)

            print(f"R2: {self._performance[0]}, RMSE: {self._performance[1]}")


        else:
            pass

    def score(self, x_test, y_true):
        """Function to retrieve model performance. Only used after applying the models training method"""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        r_score = []
        rmse = []
        for model in self._model:
            y_pred = model.predict(x_test)
            r_score.append(r2_score(y_true, y_pred))
            rmse.append(root_mean_squared_error(y_true, y_pred))

        r_score = np.mean(np.stack(r_score), axis=0)
        rmse = np.mean(np.stack(rmse), axis=0)

        return r_score, rmse

    def predict(self, x_pred):

        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        if self._model_type in ["svm", "rf"]:
            ensemble_results = []
            final_results = []
            for model in self._model:
                model_results = []
                for x in x_pred:
                    if isinstance(x, torch.Tensor):
                        x = x.detach().cpu().numpy()
                        if len(x.shape) >= 2:
                            x = x.reshape(-1)

                    model_results.append(self._model.predict(x))
                ensemble_results.append(model_results)
            for i in range(len(x_pred)):
                final_results.append(np.mean([model[i] for model in ensemble_results]))

            return final_results

        else:
            warnings.warn(
                "Model type not supported yet. Please use either 'svm' or 'rf' as model type. Sryyyyyyyyy.... ")
            return

    def get_performance(self):
        return self._performance


if __name__ == "__main__":
    pass
