import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DefogDataset(Dataset):
    def __init__(self, data_dir, file_list, window=200, step=50):
        self.samples = []
        self.labels = []

        for fname in file_list:
            df = pd.read_csv(os.path.join(data_dir, fname))

            x = df[["AccV", "AccML", "AccAP"]].values

            # look into this target more
            y_raw = df["Task"].astype(int).values
            # simple binary label for hesitation
            # if "StartHesitation" in df.columns:
            #     y = df["StartHesitation"].values.astype(np.int64)
            y = (y_raw > 0).astype(int)

            n = len(x)
            for start in range(0, n - window, step):
                segment = x[start: start + window]
                label = int(y[start: start + window].max())


                self.samples.append(segment)
                self.labels.append(label)


        self.samples = torch.tensor(self.samples, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]