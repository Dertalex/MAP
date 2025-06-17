import gc
import os.path
import random
import warnings
from datetime import datetime
from typing import Literal, Optional
from joblib import parallel_backend
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from lightgbm import early_stopping
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden_dims):
        super(NeuralNetwork, self).__init__()


class ActivityPredictor:
    _is_hypertuned = False
    _early_stopping = False
    _model_type = None
    _model = None
    _params = dict()
    _split = (80, 10, 10)
    _data_prepared = None
    _train_data = None
    _test_data = None
    _val_data = None
    _performance = None
    _roc_auc_score = None
    _is_trained = False
    _seed = None
    _data_raw = None

    def __init__(self, model_type: Literal[
        "svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge", "lasso"], x_arr, y_arr,
                 split=(80, 10, 10), params: Optional[dict] = dict, early_stopping: Optional[int] = False,
                 shuffle_data: Optional[bool] = True,
                 seed: Optional[int] = random.seed):
        self._model_type = model_type
        self._split = split
        self._early_stopping = early_stopping
        self.seed = seed
        data = [(x, y) for x, y in zip(x_arr, y_arr)]
        if shuffle_data:
            random.shuffle(data)
        self._define_model()
        self._params = params
        self._data_raw = self._split_data(data)

    def _split_data(self, data):
        train_size = int(len(data) * self._split[0] / sum(self._split))
        test_size = int(len(data) * self._split[1] / sum(self._split))

        train_data = data[:train_size]
        test_data = data[train_size:train_size + test_size]
        val_data = data[train_size + test_size:]

        splitted_data = {"x_train": [embedding[0] for embedding in train_data],
                         "x_val": [embedding[0] for embedding in val_data],
                         "x_test": [embedding[0] for embedding in test_data],
                         "y_train": [float(label[1]) for label in train_data],
                         "y_val": [float(label[1]) for label in val_data],
                         "y_test": [float(label[1]) for label in test_data]
                         }

        return splitted_data

    def _define_model(self):
        if self._model_type == "svr":
            self._model = SVR(**self._params)
        elif self._model_type == "rf":
            self._model = RandomForestRegressor(**self._params)
        elif self._model_type == "adaboost":
            self._model = AdaBoostRegressor(**self._params)
        elif self._model_type == "lightgbm":
            self._model = lgb.LGBMRegressor(**self._params)
        elif self._model_type == "xgboost":
            self._model = XGBRegressor(**self._params, early_stopping_rounds=self._early_stopping) \
                if self._early_stopping else XGBRegressor(**self._params)
        elif self._model_type == "xgboost_rf":
            self._model = XGBRFRegressor(**self._params)
        elif self._model_type == "linear":
            self._model = LinearRegression(**self._params)
        elif self._model_type == "ridge":
            self._model = Ridge(**self._params)
        elif self._model_type == "lasso":
            self._model = Lasso(**self._params)

    def train(self, k_folds: Optional[int] = 0):
        if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge",
                                "lasso"]:
            # train the model like a scikit-learn model

            # defining the inputs(x) and their label(y)
            x_train = self._data_raw["x_train"]
            x_val = self._data_raw["x_val"]
            x_test = self._data_raw["x_test"]

            y_train = self._data_raw["y_train"]
            y_val = self._data_raw["y_val"]
            y_test = self._data_raw["y_test"]

            # retrieve encoding type from tensor dimension
            # converting the datasets in torch.Tensor-Format to numpy arrays
            # flatten the tensors either by view (pytorch-tensor) or reshape (numpy.ndarrays)

            if isinstance(x_train[0], torch.Tensor):
                x_train = [tensor.to(dtype=torch.float32).detach().cpu().numpy() for tensor in x_train]
                x_val = [tensor.to(dtype=torch.float32).detach().cpu().numpy() for tensor in x_val]
                x_test = [tensor.to(dtype=torch.float32).detach().cpu().numpy() for tensor in x_test]

            if len(x_train[0].shape) >= 2:
                x_train = [x.reshape(-1) for x in x_train]
                x_val = [x.reshape(-1) for x in x_val]
                x_test = [x.reshape(-1) for x in x_test]

                """Tbd: other tensor input data types"""

            # create k-fold splits of dataset
            if not k_folds == "loo" and not isinstance(k_folds, int):
                raise ValueError("k_folds must be an integer or 'loo' for leave-one-out cross-validation")

            if k_folds != 0 or k_folds == "loo":

                x_train = np.concatenate([x_train, x_test])
                y_train = np.concatenate([y_train, y_test])

                self._data_prepared = {"x_train": x_train,
                                       "x_val": x_val,
                                       "y_train": y_train,
                                       "y_val": y_val,
                                       }

                model_ensemble = []

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
                    if self._model_type == "xgboost" and self._early_stopping:
                        self._model.fit(x_train_fold, y_train_fold, eval_set=[(x_test_fold, y_test_fold)])

                    elif self._model_type == "lightgbm" and self._early_stopping:
                        self._model.fit(x_train_fold, y_train_fold, eval_set=[(x_test_fold, y_test_fold)],
                                        eval_metric="rmse",
                                        callbacks=[lgb.early_stopping(stopping_rounds=self._early_stopping)])

                    else:
                        with parallel_backend("threading", n_jobs=5):
                            self._model.fit(x_train_fold, y_train_fold)

                    model_ensemble.append(self._model)

                self._model = model_ensemble

            else:  # k_folds == 0
                self._data_prepared = {"x_train": x_train,
                                       "x_val": x_val,
                                       "x_test": x_test,
                                       "y_train": y_train,
                                       "y_val": y_val,
                                       "y_test": y_test
                                       }

                self._define_model()

                if self._early_stopping is not False:
                    if self._model_type == "xgboost":
                        self._model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
                    elif self._model_type == "lightgbm":
                        self._model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l1",
                                        callbacks=[lgb.early_stopping(stopping_rounds=self._early_stopping)])
                    else:
                        with parallel_backend("threading", n_jobs=5):
                            warnings.warn(
                                "Early Stopping (es) is only supported for xgboost and lightgbm. "
                                "Scikit-RF already applies validation during training and therefore not requires es. "
                                "Other models are not supported yet and will be trained normally without es.")
                            self._model.fit(x_train, y_train)

                else:  # those models are sped up by providing "njobs" in the params:
                    if self._model_type in ["xgboost", "lightgbm"]:
                        self._model.fit(x_train, y_train)
                    else:
                        with parallel_backend("threading", n_jobs=5):
                            self._model.fit(x_train, y_train)

        self._is_trained = True
        self._performance = self.score(x_val, y_val)

        # print(f"R2: {self._performance[0]}, RMSE: {self._performance[1]}")

    def predict(self, x_pred: list, average_fold_results=True) -> list:
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge",
                                "lasso"]:
            y_pred = []
            ensemble_results = []

            if isinstance(self._model,
                          list):  # if self._model is stored as a list, it is because a cross validation approach is used with multiple models)
                models = self._model
            else:  # make it a list to iterate over its instances, even its only one model
                models = [self._model]

            for x in x_pred:

                ensemble_results = []
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                if len(x.shape) >= 2:
                    x = np.ravel(x)

                for model in models:
                    if self._early_stopping is not False:
                        if self._model_type == "xgboost":
                            result = model.predict([x], iteration_range=(0, model.best_iteration + 1))
                        elif self._model_type == "lightgbm":
                            result = model.predict([x], num_iteration=model.best_iteration_)
                        else:
                            result = model.predict([x])
                    else:
                        result = model.predict([x])
                        if isinstance(result, list):
                            result = result[0]
                    ensemble_results.append(result)
                if average_fold_results or len(ensemble_results) == 1:
                    y_pred.append(np.mean(ensemble_results))
                else:
                    y_pred.append(ensemble_results)

            return y_pred

        else:
            warnings.warn(
                "Model type not supported yet. Please use either 'svm' or 'rf' as model type. Sryyyyyyyyy.... ")
        return

    def score(self, x_val, y_val):
        """Function to retrieve model performance. Only used after applying the models training method"""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        r_score = []
        rmse = []

        y_pred = self.predict(x_val)

        r_score.append(r2_score(y_val, y_pred))
        rmse.append(root_mean_squared_error(y_val, y_pred))

        r_score = np.mean(np.stack(r_score), axis=0)
        rmse = np.mean(np.stack(rmse), axis=0)

        return r_score, rmse

    def get_performance(self):
        return self._performance

    def get_model(self):
        return self._model

    def set_model(self, new_model, is_trained=Literal[True, False]):
        if self._is_trained:
            self._model = new_model
            self._is_trained = is_trained
            self.score(x_val=self._data_prepared["x_val"], y_val=self._data_prepared["y_val"])
        else:
            warnings.warn(Warning("Train the model first before replacing the trained model - or create a new model"))

    def get_data(self, prepared: Literal[True, False]):
        if prepared:
            if self._is_trained:
                return self._data_prepared
            else:
                warnings.warn(Warning("Data has not been prepared for Training. This happens during self.train()"))
        else:
            return self._data_raw

    def save_model(self, filename):
        if len(self._model) == 1:
            ensemble = False
        else:
            ensemble = True

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if len(filename.split("/")) == 1:
            out_path = f"./models/{time_stamp}"
        else:
            out_path = "/".join(filename.split("/")[:-1])
            filename = filename.split("/")[-1]
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        outfile = os.path.join(out_path, filename)
        if self._is_trained is False:
            raise ValueError("Model has not been trained yet. Train it with the train() method before saving.")

        else:
            if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "linear", "ridge", "lasso"]:
                import pickle
                if not ensemble:
                    with open(f'{outfile}.pkl', 'wb') as f:
                        pickle.dump(self._model, f)
                if ensemble:
                    for i, model in enumerate(self._model):
                        with open(f'{outfile}_{i}.pkl', 'wb') as f:
                            pickle.dump(model, f)

            if self._model_type in ["xgboost", "xgboost_rf"]:

                if not ensemble:
                    self._model.save_model(outfile)
                else:
                    for i, model in enumerate(self._model):
                        model.save_model(f"{outfile}_{i}.json")
                    print(f'model saved to {out_path}')

    def load_model_weights(self):
        pass
