# https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

import numpy as np
import pandas as pd
import os
import torch

def fog_file(filepath):
    cols = ["time_ms", "ankle_forward", "ankle_vertical",
            "thigh_forward", "thigh_vertical", "thigh_lateral",
            "trunk_forward", "trunk_vertical", "trunk_lateral",
            "annotation"]
    

    df = pd.read_csv(filepath, sep="\\s+", header=None, names=cols)
    return df

def load_dataset(folder_path):
    all_data = {}
    for filename in os.listdir(folder_path):
        # print(filename)
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            all_data[filename] = fog_file(path)
    return all_data

# dataset = load_dataset("/users/charles/desktop/prefog-ml/testing-period/dataset/")
so_dummy_data_acc = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/acceleration.csv")
so_dummy_data_gy = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/gyro.csv")
so_dummy_data_gcd = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/GaitCycleData.csv")
so_dummy_data_mag = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/metadata.csv")
so_dummy_data_pre = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/pressure.csv")
so_dummy_data_pre = pd.read_csv("/users/charles/desktop/prefog-ml/testing-period/stappone_dummy_data/groundcontact.csv")


# print(dataset["S01R01.txt"].head())

print(so_dummy_data_acc.head())


#   time_ms  ankle_forward  ankle_vertical  thigh_forward  thigh_vertical  thigh_lateral  trunk_forward  trunk_vertical  trunk_lateral  annotation
# 15       70             39            -970              0               0              0              0               0              0           0
# 31       70             39            -970              0               0              0              0               0              0           0
# 46       60             49            -960              0               0              0              0               0              0           0
# 62       60             49            -960              0               0              0              0               0              0           0
# 78       50             39            -960              0               0              0              0               0              0           0

