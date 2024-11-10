import os
import pandas as pd
import numpy as np
import json
import copy
from utils import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from utils import calculate_nmi, nmi_sensi

class DPKMeans_RowSplit:
    def __init__(self, epsilon=1.0):
        self.max_iter = None
        self.epsilon = epsilon
        self.epsilon_iter = None
        self.n_clusters = 2
        self.c_L = None
        self.c_R = None
        self.S_L = None
        self.S_R = None

    def calc_max_iter(self, table, rho=0.225):
        n_samples = table.shape[0]
        n_dims = table.shape[1]
        epsilon_m = np.sqrt(500 * (self.n_clusters ** 3) / (n_samples ** 2) *
                            (n_dims + np.cbrt(4 * n_dims * (rho ** 2))) ** 3)

        max_iter = max(min(self.epsilon / epsilon_m, 7), 2)
        self.max_iter = int(max_iter)
        self.epsilon_iter = self.epsilon / self.max_iter

    def initialize_partition(self, table):
        n_samples = table.shape[0]
        row_indices = table.df.index
        shuffled_indices = np.random.permutation(row_indices)
        self.S_L = shuffled_indices[:n_samples // 2]
        self.S_R = shuffled_indices[n_samples // 2:]

    def fit(self, table):
        self.calc_max_iter(table)
        self.initialize_partition(table)

        for i in range(self.max_iter):
            self.update_centers(table)
            self.update_partitions(table)

    def update_centers(self, table):
        table_L = table.get_subtable(self.S_L, axis=0)
        table_R = table.get_subtable(self.S_R, axis=0)

        self.c_L = table_L.get_table_center()
        self.c_R = table_R.get_table_center()

    def update_partitions(self, table):
        n_sampels = table.shape[0]
        dist_L = table.calc_dist(self.c_L)
        dist_R = table.calc_dist(self.c_R)
        diff = dist_L - dist_R

        laplace_scale = table.dist_sensi() / self.epsilon_iter

        diff_noisy = diff + np.random.laplace(0, laplace_scale, len(diff))
        diff_noisy = pd.to_numeric(diff_noisy, errors='coerce').fillna(0)

        self.S_L = diff_noisy.nlargest(n_sampels // 2).index
        self.S_R = diff_noisy.index.drop(self.S_L)


class Table:
    def __init__(self, df, metadata, config):
        self.df = df[metadata["attributes"]]
        self.foreign_key_df = df[metadata["foreign keys"]]
        if metadata["primary keys"]:
            self.primary_key_df = df[metadata["primary keys"]]
        else:
            self.primary_key_df = None
        self.metadata = metadata
        self.numerical_scale = 1.0
        self.categorical_scale = 1.0
        self.categorical_columns, self.numerical_columns = self.categorical_numerical_division(self.columns)
        self.transformed_categorical_df = self.transform_categorical()
        self.transformed_numerical_df = self.transform_numerical()
        self.hamming_weights_df = self.calc_hamming_weights()
        self.categorical_distance = config["categorical_distance"]

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return self.df.columns

    @property
    def values(self):
        return self.df.values

    @property
    def all_columns(self):
        columns = self.df.columns.tolist()
        if self.primary_key_df is not None:
            columns = self.primary_key_df.columns.tolist() + columns
        columns = columns + self.foreign_key_df.columns.tolist()
        return columns

    def get_column_metadata(self, column_name):
        return self.metadata["columns"][column_name]

    def get_subtable(self, idxs, axis=0):
        subtable = copy.copy(self)
        if axis == 0:
            subtable.df = subtable.df.loc[idxs]
            subtable.foreign_key_df = subtable.foreign_key_df.loc[idxs]
            subtable.transformed_categorical_df = subtable.transformed_categorical_df.loc[idxs]
            subtable.transformed_numerical_df = subtable.transformed_numerical_df.loc[idxs]
        elif axis == 1:
            subtable.df = subtable.df.loc[:, idxs]
            subtable.categorical_columns, subtable.numerical_columns = subtable.categorical_numerical_division(
                subtable.columns)
            categorical_columns, numerical_columns = subtable.categorical_numerical_division(idxs)
            subtable.transformed_categorical_df = subtable.transformed_categorical_df.loc[:, categorical_columns]
            subtable.transformed_numerical_df = subtable.transformed_numerical_df.loc[:, numerical_columns]
            subtable.hamming_weights_df = subtable.hamming_weights_df.loc[:, categorical_columns]

        return subtable

    def get_table_center(self):
        c_numerical = self.transformed_numerical_df.mean(axis=0)
        if not self.transformed_categorical_df.empty:
            c_categorical = self.transformed_categorical_df.mode().iloc[0]
        else:
            c_categorical = None
        center = pd.concat([c_numerical, c_categorical])
        # to_concat = [c_numerical, c_categorical]
        # to_concat = [df for df in to_concat if not df.empty]
        # center = pd.concat(to_concat)
        return center

    def categorical_numerical_division(self, column_names):
        categorical_columns = []
        numerical_columns = []
        for column_name in column_names:
            vtype = self.get_column_metadata(column_name)["vtype"]
            if vtype == "Categorical":
                categorical_columns.append(column_name)
            elif vtype == "Integer" or vtype == "Float":
                numerical_columns.append(column_name)
            else:
                raise ValueError("Variable type not supported.")
        return categorical_columns, numerical_columns

    def column_hist(self, column_name, foreign=False):
        if foreign:
            column = self.foreign_key_df[column_name]
        else:
            column = self.df[column_name]
        column_metadata = self.get_column_metadata(column_name)
        if "values_to_count" in column_metadata.keys():
            values_to_count = column_metadata["values_to_count"]
            hist = column[column.isin(values_to_count)].value_counts().reindex(values_to_count, fill_value=0)
        elif "bins" in column_metadata.keys():
            bins = column_metadata["bins"]
            hist = pd.cut(column, bins=bins, right=False).value_counts().sort_index()
        elif "lists_to_count" in column_metadata.keys():
            value_lists = column_metadata["lists_to_count"]
            hist = pd.Series(dtype=int)
            for value_list in value_lists:
                count = column.isin(value_list).sum()
                hist[str(value_list)] = count
        else:
            raise ValueError("X-axis of histogram not defined.")

        return hist

    def transform_categorical(self):
        encoder = LabelEncoder()
        transformed_df = pd.DataFrame(index=self.df.index)
        for column_name in self.categorical_columns:
            transformed_df[column_name] = encoder.fit_transform(self.df[column_name])
        return transformed_df

    def calc_hamming_weights(self):
        hamming_weights = []
        for column_name in self.categorical_columns:
            # Error: division by zero
            unique_count = self.transformed_categorical_df[column_name].nunique()
            # if unique_count == 1:
            #     unique_count = 1+1e-10
            hamming_weight = 1 / (unique_count - 1)
            hamming_weights.append(hamming_weight)
        df = pd.DataFrame(data=[hamming_weights], columns=self.categorical_columns)
        return df

    def transform_numerical(self):
        scaler_minmax = MinMaxScaler(feature_range=(0.0, self.numerical_scale))
        transformed_df = pd.DataFrame(index=self.df.index)
        for column_name in self.numerical_columns:
            column_metadata = self.get_column_metadata(column_name)
            if "scaler" in column_metadata.keys():
                scaler_name = column_metadata["scaler"]
                if scaler_name == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_name == "PowerTransformer":
                    scaler = PowerTransformer()
                elif scaler_name == "RobustScaler":
                    scaler = RobustScaler()
                else:
                    raise ValueError("Unsupported scaler.")
                transformed_df[column_name] = scaler.fit_transform(self.df[[column_name]]).ravel()
                transformed_df[column_name] = scaler_minmax.fit_transform(transformed_df[[column_name]]).ravel()
            else:
                transformed_df[column_name] = scaler_minmax.fit_transform(self.df[[column_name]]).ravel()

        return transformed_df

    def calc_dist(self, center):
        numerical_dist = self.calc_numerical_dist(center)
        categorical_dist = self.calc_categorical_dist(center)
        return numerical_dist + categorical_dist

    def dist_sensi(self):
        numerical_sensi = self.numerical_scale * len(self.numerical_columns)
        if self.categorical_distance == "weighted_hamming":
            categorical_sensi = self.categorical_scale * self.hamming_weights_df.sum(axis=1)[0]
        elif self.categorical_distance == "hamming":
            categorical_sensi = self.categorical_scale * len(self.categorical_columns)
        else:
            raise ValueError("Unsupported categorical distance.")

        return numerical_sensi + categorical_sensi

    def calc_numerical_dist(self, center):
        numerical_center = center[self.numerical_columns]
        dist = self.transformed_numerical_df.sub(numerical_center).abs().sum(axis=1)
        return dist

    def calc_categorical_dist(self, center):
        categorical_center = center[self.categorical_columns]
        if self.categorical_distance == "weighted_hamming":
            dist = self.transformed_categorical_df.ne(categorical_center)
            dist *= self.hamming_weights_df
        elif self.categorical_distance == "hamming":
            dist = self.transformed_categorical_df.ne(categorical_center)
        else:
            raise ValueError("Unsupported categorical distance.")
        dist = dist.sum(axis=1)
        return dist * self.categorical_scale

    @staticmethod
    def rowSplit(table, epsilon, beta=None):
        dp_kmeans = DPKMeans_RowSplit(epsilon=epsilon)
        dp_kmeans.fit(table)
        S_L, S_R = dp_kmeans.S_L, dp_kmeans.S_R

        if beta:
            if len(S_L) < beta:
                ele_nb = beta - len(S_L)
                idxs = np.random.choice(len(S_R), ele_nb, replace=False)
                selected_ele = S_R[idxs]
                S_L = np.concatenate((S_L, selected_ele))
                S_R = np.delete(S_R, idxs)
            elif len(S_R) < beta:
                ele_nb = beta - len(S_R)
                idxs = np.random.choice(len(S_L), ele_nb, replace=False)
                selected_ele = S_L[idxs]
                S_R = np.concatenate((S_R, selected_ele))
                S_L = np.delete(S_L, idxs)

        return S_L, S_R

    @staticmethod
    def colSplit(table, epsilon):
        nmi_list = []
        candidate_list = []
        num_cols = table.shape[1]
        column_names = table.columns

        if epsilon == 0.0:
            shuffled_column_names = np.random.permutation(column_names)
            S_L = shuffled_column_names[:num_cols // 2]
            S_R = shuffled_column_names[num_cols // 2:]
            return S_L, S_R

        for k in range(num_cols):
            shuffled_column_names = np.random.permutation(column_names)
            S_L = shuffled_column_names[:num_cols // 2]
            S_R = shuffled_column_names[num_cols // 2:]

            nmi = calculate_nmi(table, (S_L, S_R))
            nmi_list.append(nmi)
            candidate_list.append((S_L, S_R))

        nmis = np.array(nmi_list)

        weights = np.exp(- nmis * epsilon / (2 * nmi_sensi(table.shape[0])))
        if np.all(weights == 0):
            weights = np.ones_like(weights)

        probabilities = weights / np.sum(weights)

        selected_index = np.random.choice(len(candidate_list), p=probabilities)
        selected_partition = candidate_list[selected_index]

        return selected_partition


def example_table(config):
    data = {
        "X1": [1, 2, 3, 4, 5, 6, 2],
        "X2": ["A", "B", "C", "B", "C", "A", "C"],
        "F1": [11, 22, 33, 22, 33, 11, 11]
    }
    df = pd.DataFrame(data)
    metadata = {
        "name": "table 1",
        "columns":{
            "X1": {
                "vtype": "Integer", # "Integer", "Float", "Categorical"
                # "bins": [1, 3, 5, 7],-> [1, 3) [3, 5) [5, 7)
                # "lists_to_count": [[1, 2], [3, 4], [5, 6]],
                "values_to_count": [1, 2, 3, 4, 5, 6, 7, 8],
                "scaler": "PowerTransformer"
            },
            "X2": {
                "vtype": "Categorical",
                "values_to_count": ["A", "B", "C"]
            },
            "F1": {
                "reference": "table 2",
                "vtype": "Integer",
                "values_to_count": [11, 22, 33],
            },
        },
        "attributes": ["X1", "X2"],
        "foreign keys": ["F1"],
        "tau": 3
    }
    table = Table(df, metadata, config)

    return table


def read_df_dict(table_mapping, config):
    tables = {}

    for table_name, files in table_mapping.items():
        data_file = files['data_file']
        metadata_file = files['metadata_file']

        if os.path.isfile(data_file) and os.path.isfile(metadata_file):
            print("Reading file:", data_file, metadata_file)
            df = pd.read_csv(data_file)
            with open(metadata_file, 'r') as file:
                metadata = json.load(file)
            table = Table(df, metadata, config)
            tables[table_name] = table
        else:
            raise ValueError(f"{data_file} or {metadata_file} doesn't exit")

    return tables


def adult_table(config):
    table_mapping = {
        "adult": {
            "data_file": "data/adult/adult.csv",
            "metadata_file": "data/adult/adult_domain_privbench.json"
        }
    }
    return read_df_dict(table_mapping, config)


def california_tables(config):
    table_mapping = {
        "individual": {
            "data_file": "data/California/individual.csv",
            "metadata_file": "data/California/individual_domain_PrivBench.json"
        },
        "household": {
            "data_file": "data/California/household.csv",
            "metadata_file": "data/California/household_domain_PrivBench.json"
        }
    }
    return read_df_dict(table_mapping, config)


def imdb_tables(config):
    table_mapping = {
        "title": {
            "data_file": "data/imdb/title.csv",
            "metadata_file": "data/imdb_compressed/title_domain_PrivBench.json"
        },
        "cast_info": {
            "data_file": "data/imdb/cast_info.csv",
            "metadata_file": "data/imdb_compressed/cast_info_domain_PrivBench.json"
        }
    }
    return read_df_dict(table_mapping, config)