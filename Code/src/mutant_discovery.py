import os.path
from typing import Literal, Optional
import pandas as pd
import random
import numpy as np
import warnings
import time
import csv
from scipy.stats import norm

import src.generate_encodings as ge
import src.prediction_models as pm


class Mutant_Discovery_Proteus:
    _aquisition_fn = "ei"
    _representation_type = None
    _is_discovered = False  # bool, for whether search is also proceeded or not
    _regressor = None
    _target = "maximize"
    _seed = None
    _max_seq_len = int()
    _explore = 0.1
    _wild_type = ""
    _load_from_dataframe = False
    _is_searched = False

    def __init__(self,
                 aquisition_fn: Literal["greedy", "ei", "ucb", "random"],
                 activity_predictor: pm.ActivityPredictor,
                 representation_type: Literal["one_hot", "georgiev", "blosum45", "blosum50", "blosum62", "blosum80",
                 "blosum90", "esmc_300m", "esmc_600m"],
                 wild_typ_seq,
                 sequences: [str], scores: [float],
                 explore: Optional[float] = 0.1,
                 load_from_dataframe: str = False):

        self._aquisition_fn = aquisition_fn
        self._representation_type = representation_type
        self._explore = explore
        self._wild_type = wild_typ_seq
        self._regressor = activity_predictor
        self._data = [(x, y) for x, y in zip(sequences, scores)]
        if load_from_dataframe:
            self.load_from_dataframe = load_from_dataframe

        if self._regressor._is_trained == False:
            warnings.warn("Mutant Discovrey only works with a trained regressor")

    def load_existing_discovery_df(self, file_destination) -> pd.DataFrame:
        discovery_df = pd.read_csv(file_destination)
        target_header = ["mutated_name", "mutated_seq", "y_true", "y_pred", "y_sigma", "acq_score"]
        for i, column_title in enumerate(discovery_df.head()):
            if column_title == target_header[i]:
                if column_title in ["mutated_name", "mutated_seq", "y_true", "y_pred", "y_sigma", "acq_score"]:
                    discovery_df[column_title] = discovery_df[column_title].astype(float)
                continue
            else:
                warnings.warn("Header could not be recognized to be a results file from mutant discovery. \n"
                              "Please check the file again and/or order columns as followed: \n"
                              "mutated_name, mutated_seq, y_true, y_pred, y_sigma, acq_score")

        return discovery_df

    def _get_seq_name(self, mutant_sequence):
        if len(mutant_sequence) != len(self._wild_type):
            warnings.warn("Not supporting yet name identification for sequences of different lengths.")

        seq_name = ""
        wild_type_seq = list(self._wild_type)
        for i, aa_mutant in enumerate(mutant_sequence):
            aa_wild = wild_type_seq[i]
            if aa_mutant != aa_wild:
                seq_name += f"{aa_wild}{i + 1}{aa_mutant}:"

        if seq_name == "":
            seq_name = "wildtype"
        else:
            seq_name = seq_name[:-1]  # remove last double points
        return seq_name

    def _filter_seqs(self, target):
        mean = sum([round(float(sy[1]), 4) for sy in self._data]) / len(self._data)

        if target == "maximize":
            self._data = sorted(self._data, key=lambda sy: sy[1], reverse=True)
            improved_seqs = [sy[0] for sy in self._data if float(sy[1]) > mean]
        elif target == "minimize":
            self._data = sorted(self._data, key=lambda sy: sy[1], reverse=False)
            improved_seqs = [sy[0] for sy in self._data if float(sy[1]) < mean]
        else:
            warnings.warn(f"Invalid target. Please choose between 'minimize' and 'maximize'")
            return
        return improved_seqs

    def _retrieve_mutations(self, sequences):
        if not sequences:
            return dict()
        else:

            mutations = dict()

            for i in range(len(self._wild_type)):
                differing_aa = set()
                for seq in sequences:
                    differing_aa.add(seq[i])
                if len(differing_aa) > 1:
                    mutations.update({i: list(differing_aa)})

            return mutations

    def _mutate(self, mutations_dict, explore=0.1, max_eval=100):

        if self._load_from_dataframe is not False:  # if not False, it contains the location of previous results-file
            discovery_df = self.load_existing_discovery_df(self._load_from_dataframe)
        else:
            discovery_df = pd.DataFrame(
                columns=["mutated_name", "mutated_seq", "y_true", "y_pred", "y_sigma", "acq_score"])

        i = 0
        while i < max_eval:

            seq_list = list(random.choice(self._data)[0])

            if random.random() < explore:
                # Explore: random position and random mutaion
                pos = random.randint(0, len(seq_list) - 1)
                mutation = random.choice("ACDEFGHIKLMNPQRSTVWY")

            else:
                pos = random.choice(list(mutations_dict.keys()))
                mutation = random.choice(mutations_dict[pos])

            seq_list[pos] = mutation
            mutated_seq = "".join(seq_list)
            mutated_name = self._get_seq_name(mutated_seq)

            if mutated_name not in discovery_df["mutated_name"] and mutated_seq not in discovery_df["mutated_seq"]:
                discovery_df.loc[len(discovery_df)] = [mutated_name, mutated_seq, None, None, None, None]
                i = i + 1

        return discovery_df

    def _evaluate_mutants(self, discovery_df: pd.DataFrame) -> pd.DataFrame:

        """ Acquisition Function to evaluate the predicted activity score and its resulting stadard variation
                    by Fon Funk and Laura Sofia Machado: extracted from ProteusAI acq_fn.py

        """

        def greedy(mean, std=None, current_best=None, xi=None):
            """
            Greedy acquisition function.

            Args:
                mean (np.array): This is the mean function from the GP over the considered set of points.
                std (np.array, optional): This is the standard deviation function from the GP over the considered set of points. Default is None.
                current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
                xi (float, optional): Small value added to avoid corner cases. Default is None.

            Returns:
                np.array: The mean values for all the points, as greedy acquisition selects the best based on mean.
            """
            return mean

        def EI(mean, std, current_best, xi=0.1):
            """
            Expected Improvement acquisition function.

            It implements the following function:

                    | (mu - mu^+ - xi) Phi(Z) + sigma phi(Z) if sigma > 0
            EI(x) = |
                    | 0                                       if sigma = 0

                    where Phi is the CDF and phi the PDF of the normal distribution
                    and
                    Z = (mu - mu^+ - xi) / sigma

            Args:
                mean (np.array): This is the mean function from the GP over the considered set of points.
                std (np.array): This is the standard deviation function from the GP over the considered set of points.
                current_best (float): This is the current maximum of the unknown function: mu^+.
                xi (float): Small value added to avoid corner cases.

            Returns:
                np.array: The value of this acquisition function for all the points.
            """

            Z = (mean - current_best - xi) / (std + 1e-9)
            EI = (mean - current_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
            EI[std == 0] = 0

            return EI

        def UCB(mean, std, current_best=None, kappa=1.5):
            """
            Upper-Confidence Bound acquisition function.

            Args:
                mean (np.array): This is the mean function from the GP over the considered set of points.
                std (np.array): This is the standard deviation function from the GP over the considered set of points.
                current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
                kappa (float): Exploration-exploitation trade-off parameter. The higher the value, the more exploration. Default is 0.

            Returns:
                np.array: The value of this acquisition function for all the points.
            """
            return mean + kappa * std

        def random_acquisition(mean, std=None, current_best=None, xi=None):
            """
            Random acquisition function. Assigns random acquisition values to all points in the unobserved set.

            Args:
                mean (np.array): This is the mean function from the GP over the considered set of points.
                std (np.array, optional): This is the standard deviation function from the GP over the considered set of points. Default is None.
                current_best (float, optional): This is the current maximum of the unknown function: mu^+. Default is None.
                xi (float, optional): Small value added to avoid corner cases. Default is None.

            Returns:
                np.array: Random acquisition values for all points in the unobserved set.
            """
            n_unobserved = len(mean)
            np.random.seed(None)
            random_acq_values = np.random.random(n_unobserved)
            return random_acq_values

        if self._aquisition_fn == "ei":
            acq = EI
        elif self._aquisition_fn == "greedy":
            acq = greedy
        elif self._aquisition_fn == "ucb":
            acq = UCB
        elif self._aquisition_fn == "random":
            acq = random_acquisition
        else:
            raise ValueError("Unknown acquisition function.")

        y_best = round(float(
            max([sy[1] for sy in self._data]) if self._target == "maximize" else min([sy[1] for sy in self._data])), 3)

        embeddings = ge.generate_sequence_encodings(method=self._representation_type,
                                                    sequences=discovery_df["mutated_seq"].tolist())
        y_preds = self._regressor.predict(x_pred=embeddings, average_fold_results=False)

        if isinstance(y_preds[0], list) and len(y_preds) > 1:
            y_sigmas = [round(float(np.std(np.stack(y_pred), axis=0)), 3) for y_pred in y_preds]
        else:
            y_sigmas = [round(float(np.zeros_like(y_pred)), 3) for y_pred in y_preds]

        y_preds = [round(float(np.mean(y_pred)), 3) for y_pred in y_preds]

        # for y_pred in y_preds:
        #     if len(y_pred) > 1:  # ensemble result:
        #         y_sigma = round(float(np.std(np.stack(y_pred), axis=0)), 3)
        #
        #     else:
        #         y_sigma = round(float(np.zeros_like(y_pred)), 3)
        acq_scores = []
        for i, y_pred in enumerate (y_preds):
            acq_scores.append(acq(y_pred, y_sigmas[i], y_best))
        discovery_df["y_pred"] = y_preds
        discovery_df["y_sigma"] = y_sigmas


        discovery_df["acq_score"] = acq_scores

        return discovery_df

    def _save_to_csv(self, sequences, y_values, y_pred_values, y_sigma_values, filename, acq_scores=None):
        data = []
        names = [self._get_seq_name(seq) for seq in sequences]
        header = ["mutated_name", "mutated_seq", "y_true", "y_pred", "y_sigma"]
        if acq_scores is not None:
            header.append("acq_score")

        # Drop a csv file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for i, (name, seq, y_true, y_pred, y_sigma) in enumerate(
                    zip(names, sequences, y_values, y_pred_values, y_sigma_values)):

                row = [name, seq, y_true, y_pred, y_sigma]
                if acq_scores is not None:
                    row.append(acq_scores[i])

                writer.writerow(row)
                data.append(row)

        # drop a pandas dataframe
        df = pd.DataFrame(data, columns=header)

        return df

    def search(self, target: Literal["minimize", "maximize"] = "maximize", n_mutants: int = 10000):
        self._target = target

        # selects only those sequence with a score better than average
        improved_seqs = self._filter_seqs(self._target)

        # Retrieve mutations from all improved sequences compared to the wild type
        seq_lens = set([len(seq) for seq in improved_seqs])
        # self._max_seq_len = max(len(seq) for seq in improved_seqs)
        if len(seq_lens) > 1:  # if sequences with different length exist, allow all positions to be mutated with all aa
            aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                       "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
            mutations_dict = {i: aa_list for i in range(0, max(seq_lens) - 1)}
        else:
            mutations_dict = self._retrieve_mutations([sy[0] for sy in self._data])

        # create a Search-Dataframe to collect and document results
        timestamp = time.strftime("%d-%m-%Y_%H-%M", time.localtime(time.time()))

        file_destination = f"../Results/Mutant_Discovery/Results/"
        if not os.path.exists(file_destination):
            os.makedirs(file_destination)
        out_file = os.path.join(file_destination, f"search_results_{timestamp}.csv")

        # mutations_df = self.mutation_df(mutations, improved_seqs)
        # mutate sequences
        discovered_mutants_df = self._mutate(mutations_dict=mutations_dict, explore=self._explore, max_eval=n_mutants)

        # Predict Mutant's Activity and therefore evaluate results via acquisition function
        discovery_result_df = self._evaluate_mutants(discovered_mutants_df)
        self._is_searched = True

        return discovery_result_df
