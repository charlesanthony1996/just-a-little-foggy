import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_defog import DefogDataset
from models import Foglstm

from sklearn.model_selection import KFold

def load_loso_splits(path):
    with open(path, "r") as f:
        return json.load(f)
    
def get_model(name):
    if name == "lstm":
        return Foglstm()
    else:
        raise ValueError("Unknown model")


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0

    for x, y in loader:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", help="train the model per subject. get the subject name from the json file")
    p.add_argument("--all", action="store_true", help="train all subjects")
    p.add_argument("--model", choices=["lstm", "tcn", "wavelet"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    loso = load_loso_splits("../data/defog_loso_splits.json")


    # if training all subjects
    if args.all:
        subjects = list(loso.keys())

        for sub in subjects:
            train_files = loso[sub]["train"]
            test_files = loso[sub]["test"]

            train_ds = DefogDataset("../data/train/defog", train_files)
            test_ds = DefogDataset("../data/train/defog", test_files)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32)

            model = get_model(args.model)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            for ep in range(args.epochs):
                tl = train_epoch(model, train_loader, opt, loss_fn)
                acc = eval_epoch(model, test_loader)
                print(f"subject {sub} | epoch {ep + 1} / {args.epochs} | loss={tl:.4f} | acc={acc:.4f}")

        return
    
    if not args.subject:
        raise ValueError("you must provide a prop here")
    

    sub = args.subject
    train_files = loso[sub]["train"]
    test_files = loso[sub]["test"]


    train_ds = DefogDataset("../data/train/defog", train_files)
    test_ds = DefogDataset("../data/train/defog", test_files)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = get_model(args.model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()


    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, opt, loss_fn)
        acc = eval_epoch(model, test_loader)
        print(f"Epoch {ep+ 1}/ {args.epochs} | loss={tl:.4f} | acc={acc:.4f}")



if __name__ == "__main__":
    main()





