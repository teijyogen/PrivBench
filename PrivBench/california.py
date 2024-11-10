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
from privbench import PrivBench
import time

os.chdir('/home/zheng/privbench')

def train_California(config):
    os.chdir('/home/zheng/privbench')

    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    gamma1 = config["gamma1"]
    gamma2 = config["gamma2"]
    categorical_scale = config["categorical_scale"]
    categorical_distance = config["categorical_distance"]
    leaf_noise_type = config["leaf_noise_type"]
    nonnegative_hist_method = config["nonnegative_hist_method"]
    budget_alloc_method = config["budget_alloc_method"]

    epsilon = config["epsilon"]

    start = time.time()
    california_table_dict = california_tables(config)
    privbench = PrivBench(california_table_dict, epsilon, config)
    print(f"Learning time: {time.time() - start}s.")

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    start = time.time()
    df_dict = privbench.database_synthesis()
    for table_name, df in df_dict.items():
        # print(df)
        # file_name = f"{table_name}_PrivBench_{epsilon}_1.csv"
        # file_name = f"{table_name}_PrivBench_{alpha}_{beta}_{gamma}_{gamma1}_{gamma2}_{timestamp}.csv"
        file_name = f"{table_name}_PrivBench_{epsilon}_{timestamp}.csv"
        out_path = f"out/California/{file_name}"
        df.to_csv(out_path, index = False)
        print("Saved file:", {out_path})
    print(f"Inference time: {time.time() - start}s.")


def finetune():
    # ray.shutdown()
    ray.init()

    param_space = {
        "epsilon": tune.grid_search([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]),
        "beta": tune.grid_search([10000]),
        "alpha": tune.grid_search([0.5]),
        "gamma": tune.grid_search([0.9]),
        "gamma1": tune.grid_search([0.5]),
        "gamma2": tune.grid_search([0.5]),

        "categorical_scale": tune.grid_search([1.0]),
        "categorical_distance": tune.grid_search(["weighted_hamming"]),
        "leaf_noise_type": tune.grid_search(["Geometric"]),
        "nonnegative_hist_method": tune.grid_search(["clipping"]),
        "budget_alloc_method": tune.grid_search(["uniform"]),
        # "budget_alloc_method": tune.grid_search(["zero_for_colsplit"]),
    }
    analysis = tune.run(
        train_California,
        config=param_space,
        resources_per_trial={"cpu": 1, "gpu": 0},
        num_samples=1,
        max_concurrent_trials=96,
    )
    ray.shutdown()


def main():

    config = {
        "epsilon": 3.2,
        "beta": 10000,
        "alpha": 0.5,
        "gamma": 0.9,
        "gamma1": 0.5,
        "gamma2": 0.5,
        "categorical_scale": 1.0,
        "categorical_distance": "weighted_hamming",
        "leaf_noise_type": "Geometric",
        "nonnegative_hist_method": "clipping",
        "budget_alloc_method": "uniform",
    }

    train_California(config)


if __name__ == '__main__':
    main()