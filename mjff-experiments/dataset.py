import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from glob import glob

class MjffFogDataset(Dataset):
    def __init__(self, files, window_size=200, step_size=50, normalize=True):

        self.samples = []
        self.labels = []

        for csv_path in files:
            df = pd.read_csv(csv_path)

            # features for baseline
            features = ["AccV", "AccML", "AccAP"]
            x = df[features].values.astype(np.float32)

            # simple label
            if "StartHesitation" in df.columns:
                y = df["StartHesitation"].values.astype(np.int64)

            else:
                y = np.zeros(len(df), dtype=np.int64)

            if normalize:
                x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

            # create windows
            for start in range(0, len(x) - window_size, step_size):
                end = start + window_size
                window = x[start:end]

                # window label: event if any event occured in the window
                label = 1 if np.any(y[start:end] == 1) else 0

                self.samples.append(window)
                self.labels.append(label)

        self.samples = torch.tensor(self.samples, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)



        # self.df = pd.read_csv(csv_path)
        # self.features = ["AccV", "AccML", "AccAP"]
        # self.x = self.df[self.features].values.astype(np.float32)
        # if "StartHesitation" in self.df.columns:
        #     self.y = self.df['StartHesitation'].values.astype(np.int64)

        # else:
        #     self.y = np.zeros(len(self.df), dtype=np.int64)

        # if normalize:
        #     self.x = (self.x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0)

        # self.window_size = window_size
        # self.step_size = step_size
        # self.samples, self.labels = [], []

        # for start in range(0, len(self.x) - window_size, step_size):
        #     end = start + window_size
        #     window = self.x[start:end]
        #     label = int(np.round(np.mean(self.y[start:end])))
        #     self.samples.append(window)
        #     self.labels.append(label)

    def __len__(self):
        return len(self.samples)
    
    # def __getitem__(self, idx):
    #     x = torch.tensor(self.samples[idx], dtype=torch.float32)
    #     y = torch.tensor(self.labels[idx], dtype=torch.long)
    #     return x, y

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    

def load_single_file(folder, filename):
    return [os.path.join(folder, filename)]

def load_all_files(folder):
    return glob(os.path.join(folder, "*.csv"))

def train_test_split_files(folder, split_ratio=0.8):
    files = glob(os.path.join(folder, "*.csv"))
    split = int(len(files) * split_ratio)
    return files[:split], files[split:]
