import os
import time
import numpy as np
import pandas as pd
import ray
from ray import tune
from utils import *
from table import *
from collections import Counter
from utils import calculate_nmi, distribution_to_hist
import datetime

# config = {
#     "beta": 1000,
#     "alpha": 0.5,
#     "gamma": 0.4,
#     "gamma1": 0.2,
#     "gamma2": 0.1,
#     "categorical_scale": 1.0, # 设置为0的话row splitting就不考虑categorical data
#     "categorical_distance": "weighted hamming",
#     "leaf_noise_type": "Geometric",
#     "nonnegative_hist_method": "clipping",
#     "budget_alloc_method": "uniform"
# }


class PrivSPN:
    def __init__(self, table, epsilon, config, root=False):
        self.leaf_noise_type = config["leaf_noise_type"]
        self.nonnegative_hist_method = config["nonnegative_hist_method"]
        self.budget_alloc_method = config["budget_alloc_method"]
        table.categorical_distance = config["categorical_distance"]
        table.categorical_scale = config["categorical_scale"]
        self.config = config
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma1 = config["gamma1"]
        self.gamma2 = config["gamma2"]
        self.table_size = table.shape[0]
        next_op, epsilon_op, remaining_epsilon = self.planning(table, epsilon)
        self.parent_type = next_op
        self.epsilon_op = epsilon_op
        self.parent = self.parentGen(table, next_op, epsilon_op)
        self.left, self.right = self.childrenGen(table, next_op, self.parent, remaining_epsilon)

        if root:
            self.is_root = True
            self.primary_key_df = table.primary_key_df
            self.all_columns = table.all_columns
        else:
            self.is_root = False

    # @staticmethod
    def planning(self, table, epsilon):
        # if table.shape[0] < 2 * self.beta and table.shape[1] == 1:
        if table.shape[1] == 1:
            next_op = "LEAF"
            epsilon_op = epsilon
            remaining_epsilon = 0
            return next_op, epsilon_op, remaining_epsilon

        noisy_corr, epsilon_eval = self.corrTrial(table, epsilon)
        next_op = self.decideOP(table, noisy_corr)
        epsilon_op, remaining_epsilon = self.allocBudget(table, next_op, epsilon, epsilon_eval)

        return next_op, epsilon_op, remaining_epsilon

    # @staticmethod
    def corrTrial(self, table, epsilon):
        if table.shape[0] >= 2 * self.beta and table.shape[1] > 1:
            if self.budget_alloc_method == "half":
                epsilon_eval = epsilon * 0.5 * self.gamma1
            elif self.budget_alloc_method == "uniform":
                epsilon_eval = epsilon / sigma_uniform(table, self.beta) * self.gamma1
            elif self.budget_alloc_method == "zero_for_colsplit":
                epsilon_eval = 0.0
            else:
                raise ValueError("Unsupported budget allocation method.")
            
            epsilon_eval_colsplit = epsilon_eval * self.gamma2
            epsilon_eval_laplace = epsilon_eval - epsilon_eval_colsplit
            trial_partition = Table.colSplit(table, epsilon_eval_colsplit)
            noisy_corr = calculate_nmi(table, trial_partition) + np.random.laplace(0, nmi_sensi(
                table.shape[0]) / epsilon_eval_laplace)
        else:
            epsilon_eval = 0.0
            noisy_corr = 0.0
        return noisy_corr, epsilon_eval

    # @staticmethod
    def decideOP(self, table, noisy_corr):
        if table.shape[0] >= 2 * self.beta and table.shape[1] > 1:
            if noisy_corr <= self.alpha:
                next_op = "PRODUCT"
            else:
                next_op = "SUM"
        else:
            next_op = "PRODUCT"
        return next_op

    # @staticmethod
    def allocBudget(self, table, next_op, epsilon, epsilon_eval):
        if next_op == "PRODUCT" and table.shape[1] == 2:
            epsilon_op = 0.0
        else:
            if self.budget_alloc_method == "half":
                epsilon_op = epsilon / 2 - epsilon_eval
            elif self.budget_alloc_method == "uniform":
                epsilon_op = epsilon / sigma_uniform(table, self.beta) - epsilon_eval
            elif self.budget_alloc_method == "zero_for_colsplit":
                epsilon_op = epsilon / sigma_uniform(table, self.beta)
            else:
                raise ValueError("Unsupported budget allocation method.")
        remaining_epsilon = epsilon - epsilon_eval - epsilon_op
        return epsilon_op, remaining_epsilon

    # @staticmethod
    def parentGen(self, table, next_op, epsilon_op):
        if next_op == "PRODUCT":
            parent = self.genProdNode(table, epsilon_op)
        elif next_op == "SUM":
            parent = self.genSumNode(table, epsilon_op)
        else:
            parent = self.genLeafNode(table, epsilon_op, table.columns[0])

        return parent

    # @staticmethod
    def genProdNode(self, table, epsilon_op):
        return Table.colSplit(table, epsilon_op)

    # @staticmethod
    def genSumNode(self, table, epsilon_op):
        return Table.rowSplit(table, epsilon_op, beta=self.beta)

    # @staticmethod
    def genLeafNode(self, table, epsilon_op, column_name, foreign=False):
        # column_name = table.columns[0]
        column_metadata = table.get_column_metadata(column_name)
        hist = table.column_hist(column_name, foreign=foreign)
        if self.leaf_noise_type == "Laplace":
            noise = np.random.laplace(0, 2.0 / epsilon_op, size=hist.shape)
        elif self.leaf_noise_type == "Geometric":
            noise = geometric_noise(epsilon_op, 2.0, hist.shape[0])
        else:
            raise ValueError("Unknown noise type.")
        noisy_hist = hist + noise

        if self.nonnegative_hist_method == "shift":
            hist_min = noisy_hist.min()
            if hist_min < 0:
                noisy_hist -= hist_min
        elif self.nonnegative_hist_method == "clipping":
            noisy_hist = np.clip(noisy_hist, 0, np.inf)
        else:
            raise ValueError("Unsupported type for ensuring non-negative histogram.")

        distribution = noisy_hist / noisy_hist.sum()
        return distribution, hist.sum(), column_name, column_metadata, table.df.index

    # @staticmethod
    def childrenGen(self, table, next_op, partition, remaining_epsilon):
        if next_op == "SUM":
            left_table = table.get_subtable(partition[0], axis=0)
            right_table = table.get_subtable(partition[1], axis=0)
            epsilon_left, epsilon_right = remaining_epsilon, remaining_epsilon
            left = PrivSPN(left_table, epsilon_left, self.config)
            right = PrivSPN(right_table, epsilon_right, self.config)
        elif next_op == "PRODUCT":
            left_table = table.get_subtable(partition[0], axis=1)
            right_table = table.get_subtable(partition[1], axis=1)
            if self.budget_alloc_method == "half":
                epsilon_left = (remaining_epsilon * sigma_half(left_table, self.beta)
                                / (sigma_half(left_table, self.beta) + sigma_half(right_table, self.beta)))
            elif self.budget_alloc_method == "uniform":
                epsilon_left = (remaining_epsilon * sigma_uniform(left_table, self.beta)
                                / (sigma_uniform(left_table, self.beta) + sigma_uniform(right_table, self.beta)))
            elif self.budget_alloc_method == "zero_for_colsplit":
                epsilon_left = (remaining_epsilon * sigma_uniform(left_table, self.beta)
                                / (sigma_uniform(left_table, self.beta) + sigma_uniform(right_table, self.beta)))
            else:
                raise ValueError("Unsupported budget allocation method.")

            epsilon_right = remaining_epsilon - epsilon_left
            left = PrivSPN(left_table, epsilon_left, self.config)
            right = PrivSPN(right_table, epsilon_right, self.config)
        else:
            left = None
            right = None
        return left, right

    def data_synthesis(self):
        if self.parent_type == "LEAF":
            df = self.sampling_from_leaf()
        elif self.parent_type == "PRODUCT":
            left_df = self.left.data_synthesis()
            right_df = self.right.data_synthesis()
            if left_df.shape[0] != right_df.shape[0]:
                raise ValueError("Inconsistent sizes of left and right.")
            df = pd.concat([left_df, right_df], axis=1)
        elif self.parent_type == "SUM":
            left_df = self.left.data_synthesis()
            right_df = self.right.data_synthesis()
            df = pd.concat([left_df, right_df], axis=0, ignore_index=True)
        else:
            raise ValueError("Unknown parent type.")

        if self.is_root:
            df = pd.concat([self.primary_key_df, df], axis=1)
            df = df[self.all_columns]

        return df

    def sampling_from_leaf(self):
        distribution, size, column_name, column_metadata, row_indices = self.parent
        hist = distribution_to_hist(distribution, size)
        unique_domains = hist.index.tolist()
        counts = hist.values.tolist()

        data = []

        if "values_to_count" in column_metadata.keys():
            for unique_domain, count in zip(unique_domains, counts):
                data += [unique_domain] * count
        elif "bins" in column_metadata.keys():
            if column_metadata["vtype"] == "Integer" or column_metadata["vtype"] == "Categorical":
                for unique_domain, count in zip(unique_domains, counts):
                    count = max(count, 0)
                    if count > 0:
                        data += np.random.randint(unique_domain.left, unique_domain.right, size=count).tolist()
            elif column_metadata["vtype"]  == "Float":
                for unique_domain, count in zip(unique_domains, counts):
                    data += np.random.uniform(domain.left, domain.right, size=count).tolist()
            else:
                raise ValueError("Variable type not supported.")
        elif "lists_to_count" in column_metadata.keys():
            for unique_domain, count in zip(unique_domains, counts):
                unique_domain = json.loads(unique_domain)
                data += np.random.choice(unique_domain, size=count, replace=True).tolist()
        else:
            ValueError("X-axis of histogram not defined.")

        np.random.shuffle(data)
        df = pd.DataFrame({column_name: data})
        return df

    def construct_fanout(self, table, epsilon, foreign_key_name):
        leaf_nodes = self.find_leaf_nodes()
        leaves_for_modification = self.find_leaves_for_modification(leaf_nodes)
        for leaf in leaves_for_modification:
            leaf.left = copy.deepcopy(leaf)

            leaf.right = copy.deepcopy(leaf)
            leaf.right.parent_type = "LEAF"
            leaf.right.epsilon_op = epsilon
            subtable = table.get_subtable(leaf.parent[4], axis=0)
            leaf.right.parent = self.genLeafNode(subtable, epsilon, foreign_key_name, foreign=True)
            leaf.right.left, leaf.right.right = None, None

            leaf.parent_type = "PRODUCT"
            leaf.epsilon_op = 0.0
            leaf.parent = ([leaf.left.parent[2]], [foreign_key_name])

    def find_leaf_nodes(self):
        if self.parent is None:
            return []
        if self.left is None and self.right is None:
            return [self]

        return self.left.find_leaf_nodes() + self.right.find_leaf_nodes()

    @staticmethod
    def find_leaves_for_modification(leaf_nodes):
        column_names = [leaf.parent[2] for leaf in leaf_nodes]
        counter = Counter(column_names)
        column_name = counter.most_common(1)[0][0]

        leaves_for_modification = []
        for leaf in leaf_nodes:
            if leaf.parent[2] == column_name:
                leaves_for_modification.append(leaf)

        return leaves_for_modification


class PrivBench:
    def __init__(self, table_dict, epsilon, config):
        self.config = config
        if len(table_dict) <= 1:
            self.gamma = 1.0
        else:
            self.gamma = config["gamma"]
        self.table_dict = table_dict
        self.epsilon = epsilon
        self.n_pairs = PrivBench.calculate_n_pairs(self.table_dict)
        self.spn_dict = {}
        self.construct_SPNs()
        self.construct_fanouts()

    @staticmethod
    def calculate_n_pairs(table_dict):
        n_pairs = 0
        for table in table_dict.values():
            n_pairs += len(table.metadata["foreign keys"])
        return n_pairs

    def construct_SPNs(self):
        total_tau = 0
        for table_name, table in self.table_dict.items():
            total_tau += table.metadata["tau"]

        for table_name, table in self.table_dict.items():
            # epsilon_s_i = self.epsilon * self.gamma / table.metadata["tau"] / len(self.table_dict)
            epsilon_s_i = self.epsilon * self.gamma / total_tau
            spn = PrivSPN(table, epsilon_s_i, config = self.config, root=True)
            self.spn_dict[table_name] = spn

    def construct_fanouts(self):
        if len(self.table_dict) <= 1:
            return
        total_weighted_tau = 0
        for table_name, table in self.table_dict.items():
            total_weighted_tau += table.metadata["tau"] * len(table.metadata["foreign keys"])


        for table_name, table in self.table_dict.items():
            spn = self.spn_dict[table_name]
            if len(table.metadata["foreign keys"]) > 0:
                epsilon_f_i = self.epsilon * (1 - self.gamma) / total_weighted_tau
                for foreign_key_name in table.metadata["foreign keys"]:
                    spn.construct_fanout(table, epsilon_f_i, foreign_key_name)

    def database_synthesis(self):
        df_dict = {}
        for table_name, spn in self.spn_dict.items():
            df = spn.data_synthesis()
            df_dict[table_name] = df
            
        return df_dict


if __name__ == '__main__':
    pass




