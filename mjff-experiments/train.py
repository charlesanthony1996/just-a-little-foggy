import torch
from torch.utils.data import DataLoader
from dataset import MjffFogDataset, load_single_file, load_all_files, train_test_split_files
from models import Foglstm, FogTcn, FogWaveletCnn
import torch.nn as nn
import torch.optim as optim

# csv_path = "/users/charles/desktop/prefog-ml/testing-period/mjff-experiments/mjff-data/f9fc61ce85.csv"
# csv_path = "/users/charles/downloads/tlvmc-parkinsons-freezing-gait-prediction/events.csv"
csv_path = "/users/charles/downloads/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/ffda8fadfd.csv"
folder = "/users/charles/downloads/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/"

# just a comment

# "single", "all" or "split"
mode = "all"

# "lstm", "tcn", "wavelet"
model_type = "lstm"

if mode == "single":
    files = load_single_file(folder, "ffda8fadfd.csv")

elif mode == "all":
    files = load_all_files(folder)

elif mode == "split":
    train_files, test_files = train_test_split_files(folder)
    files = train_files


dataset = MjffFogDataset(files)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

if model_type == "lstm":
    model = Foglstm()

elif model_type == "tcn":
    model = FogTcn()

elif model_type == "wavelet":
    model = FogWaveletCnn()

# choose model here
# model = Foglstm(input_size=3, num_classes=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses = []

for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}, loss: {total_loss/len(loader):.4f}")

# plotting the loss curve
import matplotlib.pyplot as plt

plt.scatter(range(len(losses)), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("mfff-experiment-loss-curve.png")
plt.show()