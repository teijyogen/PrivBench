import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


def calculate_nmi(table, partition):
    left_table = table.get_subtable(partition[0], axis=1)
    right_table = table.get_subtable(partition[1], axis=1)
    entropy_left = calculate_entropy(left_table)
    entropy_right = calculate_entropy(right_table)
    entropy_joint = calculate_entropy(table)
    nmi = (entropy_left + entropy_right - entropy_joint) / np.log2(table.shape[0])
    return nmi


def calculate_entropy(table):
    value_counts = table.df.value_counts()
    entropy_value = entropy(value_counts)
    return entropy_value


def nmi_sensi(n):
    return ((2 / n) * np.log2((n + 1) / 2) + (1 - 1 / n) * np.log2((n + 1) / (n - 1))) / np.log2(n)


def sigma_half(table, beta):
    num_rows, num_cols = table.shape[0], table.shape[1]
    # print(f"num_cols: {num_cols}, num_rows: {num_rows}\, beta: {beta}")

    # sigma = 2 ** (num_cols + num_rows / beta - 2) * num_cols * num_rows / beta
    log_sigma = (num_cols + num_rows / beta - 2) * math.log(2) + math.log(num_cols) + math.log(num_rows) - math.log(beta)
    # print("sigma_m:", math.exp(log_sigma))
    return math.exp(log_sigma)


def sigma_uniform(table, beta):
    num_rows, num_cols = table.shape[0], table.shape[1]
    return 2 * num_cols * num_rows / beta - 1


def distribution_to_hist(distribution, size):
    hist = (distribution * size).round()
    hist = hist.fillna(0).astype(int)
    while hist.sum() != size:
        if size - hist.sum() > 0:
            hist.loc[hist.idxmax()] += 1
        else:
            hist.loc[hist.idxmax()] -= 1

        hist = hist.apply(lambda x: max(0, x))

    return hist


def geometric_noise(epsilon, sensitivity, size):
    p = 1 - np.exp(-epsilon / sensitivity)
    noise_vec = []
    for i in range(size):
        u = np.random.random()
        noise = np.floor(np.log(u) / np.log(1 - p))
        
        noise = noise if np.random.rand() > 0.5 else -noise
        noise_vec.append(noise)
    noise_vec = np.array(noise_vec)

    return noise_vec

