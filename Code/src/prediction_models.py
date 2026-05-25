import gc
import os.path
import warnings
import random
from datetime import datetime
from typing import Literal, Optional, List, Tuple
from joblib import parallel_backend
import numpy as np
import copy

from src.utils import HiddenPrints, HiddenWarnings, proper_time, make_folds
from src.metrics import *
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from torch.nn import MarginRankingLoss

from xgboost import XGBRegressor, XGBRFRegressor
import lightgbm as lgb
from lightgbm import early_stopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeedForwardNet(nn.Module):
    default_params = dict(
        input_dim=None,  # must be provided!
        hidden_dim=1000,  # default hidden size
        n_hidden_layers=4,
        dropout=0.3,
        max_epochs=200,
        learning_rate=0.001,
        batch_size=32,
        early_stopping=10,
        weight_decay=0.0,
        device="auto",
        criterion="MSE"
    )

    def __init__(self, **kwargs):
        super(FeedForwardNet, self).__init__()
        # Merge defaults with user params
        params = {**self.default_params, **kwargs}

        self._max_epochs = params["max_epochs"]
        self._learning_rate = params["learning_rate"]
        self._batch_size = params["batch_size"]
        self._early_stopping = params["early_stopping"]
        self._criterion = params["criterion"]
        if self._criterion == "Ranking" and self._batch_size % 2 != 0:
            self._batch_size = self._batch_size + 1
        if params["device"] != "auto":
            self._device = params["device"]
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._input_dim = params["input_dim"]
        self._hidden_dim = params["hidden_dim"]
        self._n_hidden_layers = params["n_hidden_layers"]
        self._dropout = params["dropout"]
        self._weight_decay = params["weight_decay"]
        # Build network dynamically
        layers = []
        in_features = self._input_dim
        for i in range(self._n_hidden_layers):
            out_features = self._hidden_dim if i == 0 else max(in_features // 2, 4)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if self._dropout > 0:
                layers.append(nn.Dropout(self._dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self._network = nn.Sequential(*layers)
        self.to(self._device)

    def _forward(self, x):
        return self._network(x)

    def predict(self, x):
        with torch.no_grad():
            model = self.to(self._device)
            model.eval()
            x = x.to(self._device)
            result = model._forward(x)
            return result

    def fit(self, train_set: Tuple[torch.Tensor, torch.Tensor], eval_set: Tuple[torch.Tensor, torch.Tensor],
            maximize: bool = False):
        loss_fn = {"MSE": torch.nn.MSELoss(),
                   "Ranking": torch.nn.MarginRankingLoss()}
        criterion = loss_fn[self._criterion]

        optimization_directions = {"MSE": False,
                                   "Ranking": False}
        maximize = optimization_directions[self._criterion]
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

        train_loader = DataLoader(TensorDataset(train_set[0], train_set[1]), batch_size=int(self._batch_size),
                                  shuffle=True)
        val_loader = DataLoader(TensorDataset(eval_set[0], eval_set[1]), batch_size=int(self._batch_size), shuffle=True)

        worsened = 0
        best_weights = copy.deepcopy(self.state_dict())
        best_epoch = 0
        best_val_loss = -999999 if maximize else 999999
        n_epochs = 0

        while worsened < self._early_stopping:
            self.train()
            total_train_loss = 0
            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(self._device), y_train.to(self._device)
                y_pred = self._forward(x_train)
                if self._criterion == "MSE":
                    train_loss = criterion(y_pred, y_train)

                if self._criterion == "Ranking":
                    y_pred_1 = y_pred[0:y_pred.shape[0]//2].reshape(-1)
                    y_pred_2 = y_pred[y_pred.shape[0]//2:].reshape(-1)
                    y_true_1 = y_train[0:y_pred.shape[0]//2].reshape(-1)
                    y_true_2 = y_train[y_pred.shape[0]//2:].reshape(-1)

                    target = torch.ones_like(y_true_1)
                    target[y_true_1 < y_true_2] = -1
                    target[y_true_1 == y_true_2] = 1
                    train_loss = criterion(y_pred_1, y_pred_2, target)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
            total_train_loss = total_train_loss / len(train_loader)

            # validation
            with torch.no_grad():
                self.eval()
                total_val_loss = 0
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self._device), y_val.to(self._device)
                    y_pred = self._forward(x_val)
                    if self._criterion == "MSE":
                        val_loss = criterion(y_pred, y_val)

                    if self._criterion == "Ranking":
                        y_pred_1 = y_pred[0:y_pred.shape[0] // 2].reshape(-1)
                        y_pred_2 = y_pred[y_pred.shape[0] // 2:].reshape(-1)
                        y_true_1 = y_val[0:y_pred.shape[0] // 2].reshape(-1)
                        y_true_2 = y_val[y_pred.shape[0] // 2:].reshape(-1)

                        target = torch.ones_like(y_true_1)
                        target[y_true_1 < y_true_2] = -1
                        target[y_true_1 == y_true_2] = 1
                        val_loss = criterion(y_pred_1, y_pred_2, target)

                    total_val_loss += val_loss.item()
                total_val_loss = total_val_loss / len(val_loader)

                improved = (total_val_loss < best_val_loss) if not maximize else (total_val_loss > best_val_loss)

                if improved:
                    worsened = 0
                    best_epoch = n_epochs
                    best_val_loss = total_val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                else:
                    worsened += 1

            n_epochs += 1

            if self._max_epochs and n_epochs >= self._max_epochs:
                print("Reached maximum epochs. Stopping training.")
                break

            print(
                f"Epoch {n_epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")

        if not self._max_epochs and worsened >= self._early_stopping:
            print(f"Early stopping after {self._early_stopping} epochs without improvement.")

        self.load_state_dict(best_weights)
        self.eval()
        print("     ------     ")
        print(f"Final Model State: epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
        print("     ======     ")


class ActivityPredictor:
    _is_hypertuned = False
    _early_stopping = False
    _model_type = None
    _model = None
    _params = dict()
    _split = (80, 10, 10)
    _train_data = None
    _test_data = None
    _val_data = None
    _performance = None
    _is_trained = False
    _seed = None
    _data = None

    def __init__(self, model_type: Literal[
        "svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge", "lasso", "elastic_net", "fnn"],
                 x_arr, y_arr,
                 split=(80, 20), params: Optional[dict] = dict, early_stopping: Optional[int] = False,
                 shuffle_data: Optional[bool] = True,
                 seed: Optional[int] = random.seed):
        self._model_type = model_type
        self._split = split
        self._early_stopping = early_stopping
        self.seed = seed
        data = [(x, y) for x, y in zip(x_arr, y_arr)]
        if shuffle_data:
            random.shuffle(data)
        self._params = params
        self._data = self._split_data(data)

    def _split_data(self, data):
        train_size = int(len(data) * self._split[0] / sum(self._split))

        train_data = data[:train_size]
        val_data = data[train_size:]

        splitted_data = {"x_train": [embedding[0] for embedding in train_data],
                         "x_val": [embedding[0] for embedding in val_data],
                         "y_train": [float(label[1]) for label in train_data],
                         "y_val": [float(label[1]) for label in val_data]
                         }

        return splitted_data

    def _define_model(self):
        if self._model_type == "svr":
            return SVR(**self._params)
        elif self._model_type == "rf":
            return RandomForestRegressor(**self._params)
        elif self._model_type == "adaboost":
            return AdaBoostRegressor(**self._params)
        elif self._model_type == "lightgbm":
            return lgb.LGBMRegressor(**self._params, n_jobs=-1)
        elif self._model_type == "xgboost":
            if self._early_stopping:
                self._params["early_stopping_rounds"] = self._early_stopping
            return XGBRegressor(**self._params, feval=spearman_xgboost, maximize=True, n_jobs=-1)
        elif self._model_type == "xgboost_rf":
            return XGBRFRegressor(**self._params)
        elif self._model_type == "linear":
            return LinearRegression(**self._params)
        elif self._model_type == "ridge":
            return Ridge(**self._params)
        elif self._model_type == "lasso":
            return Lasso(**self._params)
        elif self._model_type == "elastic_net":
            return ElasticNet(**self._params)
        elif self._model_type == "fnn":
            self._params["early_stopping"] = self._early_stopping
            return FeedForwardNet(**self._params)

    def train(self, k_folds: Optional[int] = 1):

        torch.cuda.empty_cache()
        # train the model like a scikit-learn model, prepare data for scikit learn -alike models as flattened lists/np arrays
        if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge",
                                "lasso", "elastic_net"]:
            if isinstance(self._data["x_train"][0], torch.Tensor):
                for key in ["x_train", "x_val"]:
                    self._data[key] = [tensor.detach().cpu().numpy() for tensor in self._data[key]]
            for key in self._data.keys():
                if len(np.ravel(self._data[key][0])) > 1:
                    self._data[key] = [np.ravel(x_or_y) for x_or_y in self._data[key]]
                else:
                    self._data[key] = [x_or_y for x_or_y in self._data[key]]

        # model is a feed-forward neural network, prepare data for pytorch ANN as flattened tensors
        elif self._model_type in ["fnn"]:
            if not isinstance(self._data["x_train"][0], torch.Tensor):  # convert data to torch tensors
                for key in ["x_train", "x_val"]:
                    self._data[key] = [torch.tensor(x, dtype=torch.float32) for x in self._data[key]]
                
            if not isinstance(self._data["y_train"][0], torch.Tensor):  # convert data to torch tensors
                for key in ["y_train", "y_val"]:
                    self._data[key] = [torch.tensor(x, dtype=torch.float32) for x in self._data[key]]
            
            for key in self._data.keys():  # flatten tensors
                self._data[key] = [tensor.reshape(-1) for tensor in self._data[key]]

        # create k-fold splits of dataset
        folds = make_folds(self._data["x_train"], self._data["y_train"], k_folds=k_folds)
        if k_folds == 1:
            folds[0][2], folds[0][3] = self._data["x_val"], self._data["y_val"]

        model_ensemble = []

        for x_train, y_train, x_val, y_val in folds:
            if self._model_type == "fnn":
                self._params["input_dim"] = len(x_train[0])  # set input dimension for FNNs

            model = self._define_model()  # initiate multiple model instances for ensemble learnin

            if self._model_type == "xgboost" and self._early_stopping:
                model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

            elif self._model_type == "lightgbm" and self._early_stopping:
                model.fit(
                    X=x_train,
                    y=y_train,
                    eval_set=[(np.array(x_val), np.array(y_val))],
                    eval_metric=spearman_lightgbm,
                    callbacks=[lgb.early_stopping(stopping_rounds=self._early_stopping)]
                )

            elif self._early_stopping is not False and self._model_type != "fnn":
                    warnings.warn(
                        "Early Stopping is only supported for xgboost and lightgbm. Since the chosen model is not one of those, this parameter will be ignored.")
                    model.fit(x_train, y_train)
            else:
                if self._model_type == "xgboost":
                    model.fit(x_train, y_train)
                elif self._model_type == "lightgbm":
                    model.fit(x_train, y_train, eval_metric=spearman_lightgbm)
                elif self._model_type == "fnn":
                    model.fit(train_set=(torch.stack(x_train), torch.stack(y_train)),
                              eval_set=(torch.stack(x_val), torch.stack(y_val)))
                else:
                    with parallel_backend("threading", n_jobs=-1):
                        model.fit(x_train, y_train)

            model_ensemble.append(model)

        self._model = model_ensemble
        self._is_trained = True
        self._performance = self.score(self._data["x_val"], self._data["y_val"])

    def predict(self, x_pred: list, average_fold_results=True) -> list:
        # basic input checks

        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        # prepare input data for prediction - and check whether model type is supported
        if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "xgboost", "xgboost_rf", "linear", "ridge",
                                "lasso", "elastic_net"]:
            if isinstance(x_pred[0], torch.Tensor):
                x_pred = [tensor.detach().cpu().numpy() for tensor in x_pred]
            x_pred = [np.ravel(embedding) for embedding in x_pred]

        elif self._model_type in ["fnn"]:
            if not isinstance(x_pred[0], torch.Tensor):
                x_pred = [torch.tensor(embedding, dtype=torch.float32) for embedding in x_pred]
            x_pred = torch.stack([embedding.reshape(-1) for embedding in x_pred])
            torch.cuda.empty_cache()

        else:
            warnings.warn(
                "Model type not supported yet. Please use either 'svm' or 'rf' as model type. Sryyyyyyyyy.... ")
            return

        gc.collect()

        y_pred = []
        models = self._model  # model(s) always stored as list for ensemble learning

        for model in models:
            if self._early_stopping is not False and self._model_type == "xgboost":
                result = model.predict(x_pred, iteration_range=(0, model.best_iteration + 1))
            elif self._early_stopping is not False and self._model_type == "lightgbm":
                result = model.predict(x_pred, num_iteration=model.best_iteration_)

            else:
                result = model.predict(x_pred)
                if self._model_type in ["fnn"]:
                    result = result.detach().cpu().numpy().reshape(-1)

            y_pred.append(result)

        if average_fold_results:
            return np.mean(y_pred, axis=0)
        else:
            return y_pred

    def score(self, x_val, y_val):
        """Function to retrieve model performance. Only used after applying the models training method"""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        ndcg = []
        spearman = []
        pearson = []
        r_squared = []
        meansquarederror = []

        y_pred = self.predict(x_val) #output already properly

        if isinstance(y_val[0], torch.Tensor):
            y_val = [y.detach().cpu().numpy() for y in y_val]
        if isinstance(y_val[0], np.ndarray):
            y_val = [float(y.item()) if y.size == 1 else float(y.flat[0]) for y in y_val]
        ndcg.append(ndcg_score(y_pred, y_val))
        pearson.append(pearson_correlation(y_pred, y_val))
        spearman.append(spearman_correlation(y_pred, y_val))
        r_squared.append(r2_score(y_pred, y_val))
        meansquarederror.append(mse(y_pred, y_val))

        ndcg = np.mean(np.stack(ndcg), axis=0)
        pearson = np.mean(np.stack(pearson), axis=0)
        spearman = np.mean(np.stack(spearman), axis=0)
        r_squared = np.mean(np.stack(r_squared), axis=0)
        meansquarederror = np.mean(np.stack(meansquarederror), axis=0)

        return ndcg, spearman, pearson, r_squared, meansquarederror

    def get_performance(self):
        return self._performance

    def get_model(self):
        return self._model

    def set_model(self, new_model, is_trained=Literal[True, False]):
        if self._is_trained:
            self._model = new_model
            self._is_trained = is_trained
            self.score(x_val=self._data["x_val"], y_val=self._data["y_val"])
        else:
            warnings.warn(Warning("Train the model first before replacing the trained model - or create a new model"))

    def get_data(self):
        return self._data

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
            if self._model_type in ["svr", "rf", "adaboost", "lightgbm", "linear", "ridge", "lasso", "elastic_net"]:
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

            if self._model_type in ["fnn"]:
                if not ensemble:
                    torch.save(self._model.state_dict(), f"{outfile}.pt")
                else:
                    for i, model in enumerate(self._model):
                        torch.save(model.state_dict(), f"{outfile}_{i}.pt")
                print(f'model saved to {out_path}')

    def load_model_weights(self):
        pass
