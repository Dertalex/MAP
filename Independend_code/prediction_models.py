from typing import Literal
import pandas as pd
import random
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.svm import SVR
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
    _is_trained = False
    _seed = None

    def __init__(self, model_type: Literal["rf", "svm"], data, x_column_index: int, y_column_index: int,
                 split=(80, 10, 10), seed=None):
        self._model_type = model_type
        self._split = split
        self._data = data
        self._x_column = x_column_index
        self._y_column = y_column_index
        self.seed: None
        self.define_model()
        self.split_data()

    def split_data(self):
        # check if data has headers

        has_header = False if self._data[0][self._y_column].isnumeric() else True
        data = self._data[1:] if has_header else self._data

        if self._seed:
            random.seed(self._seed)
            random.shuffle(data)

        len_data = len(data) if has_header == False else len(data) - 1
        train_split = int(len_data * self._split[0] / 100)
        valid_split = int(len_data * self._split[1] / 100)

        self._train_data = data[:train_split]
        self._val_data = data[train_split:train_split + valid_split]
        self._test_data = data[train_split + valid_split:]

    def define_model(self):
        models = {"svm": SVR(), "rf": RandomForestRegressor()}
        self._model = models[self._model_type]

    def train_scikit_model(self):

        # defining the inputs(x) and their label(y)
        x_train = [encoding[self._x_column] for encoding in self._train_data]
        x_val = [encoding[self._x_column] for encoding in self._val_data]
        x_test = [encoding[self._x_column] for encoding in self._test_data]

        y_train = [label[self._y_column] for label in self._train_data]
        y_val = [label[self._y_column] for label in self._val_data]
        y_test = [label[self._y_column] for label in self._test_data]

        # retrieve encoding type from tensor dimensions
        """To Fix: how to multidimensional tensors? (3D and higher)"""

        # converting the datasets in torch.Tensor-Format to numpy arrays
        # flatten the tensors either by view (pytorch-tensor) or reshape (numpy.ndarrays)
        x_sets = [x_train, x_val, x_test]

        if isinstance(x_train[0], torch.Tensor):
            x_train = [tensor.detach().cpu().numpy() for tensor in x_train]
            x_val = [tensor.detach().cpu().numpy() for tensor in x_val]
            x_test = [tensor.detach().cpu().numpy() for tensor in x_test]

        if len(x_train[0].shape) >= 2:
            x_train = [x.reshape(-1) for x in x_train]
            x_val = [x.reshape(-1) for x in x_val]
            x_test = [x.reshape(-1) for x in x_test]

            """To Fix: other tensor input data types"""

        x_train.extend(x_val)
        y_train.extend(y_val)

        # train the model
        self._model.fit(x_train, y_train)

        # determine the performance of the model

        self._is_trained = True
        self._performance = self._score(x_test, y_test)

        print(self._performance)


    def _score(self, x_test, y_true):
        """Function to retrieve model performance. Only used after applying the models training method"""

        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Train it with the train() method")

        y_pred = self._model.predict(x_test)

        # for embedding in x_test:
        #     y_pred.append(self._model.predict(embedding))

        r_score = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        return f"R^2 : {r_score}, RSME {rmse}"


def get_performance(self):
    return self._performance


if __name__ == "__main__":
    pass
