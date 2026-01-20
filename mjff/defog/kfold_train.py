import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataset_defog import DefogDataset
from models import Foglstm, FogTcn
from splits_kfold import build_group_kfold
from train_eval import train_epoch, eval_epoch
from metrics import compute_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_model(name):
    if name == "lstm":
        return Foglstm()
    if name == "tcn":
        return FogTcn()
    raise ValueError("unknown model")

def run_kfold(subject_files, data_dir, model_name="lstm", k=5, epochs=10, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = build_group_kfold(subject_files, k=k, seed=seed)
    results = []

    for f in folds:
        print(f"\n=== Fold {f['fold'] + 1}/{k} ===")

        # 🔁 WEIGHTS RESET HERE
        set_seed(seed + f["fold"])
        model = get_model(model_name).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # ----- Train / Val split (10% of train files)
        train_files = f["train_files"]
        split = int(0.9 * len(train_files))
        tr_files = train_files[:split]
        val_files = train_files[split:]

        train_ds = DefogDataset(data_dir, tr_files)
        val_ds = DefogDataset(data_dir, val_files)
        test_ds = DefogDataset(data_dir, f["test_files"])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)

        # ----- Training
        for ep in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            print(f"epoch {ep+1}/{epochs} | train_loss={loss:.4f}")

        # ----- Validation (optional monitoring)
        yv, pv = eval_epoch(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, num_classes=2)

        # ----- Testing
        yt, pt = eval_epoch(model, test_loader, device)
        test_metrics = compute_metrics(yt, pt, num_classes=2)

        print("TEST METRICS:", test_metrics)

        results.append({
            "fold": f["fold"],
            **test_metrics
        })

    return pd.DataFrame(results)
