import src.generate_encodings as ge
import src.prediction_models as pm
import tqdm
import os, sys
from joblib import parallel_backend
import ast
import sys
import torch


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


"""load encodings"""
pass
y_span = 2.4
min_bin_size = 0.005
max_bin_size = 0.01
bin_samples = 5

""" XGB HyperParameter"""
iterations = []
learning_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
minimum_loss = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
max_depths = [4, 5, 6, 7, 8, 9, 10]
tree_methods = ["hist", "approx", "auto"]
max_bins = list(range(int(y_span / max_bin_size), int(y_span / min_bin_size) + 1, 40))

""" RF HyperParameter"""
x_val = rf_model.get_data()["x_val"]
y_val = rf_model.get_data()["y_val"]
