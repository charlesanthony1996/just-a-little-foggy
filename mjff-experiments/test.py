import os
import zipfile
import pandas as pd

# DATASET = "charlesanthony1996/mjff-train-defog-folder"
DATASET = "charlesanthony1996/stappone-dummy-data"
# DATASET = "charlesanthony1996/daphnet-data"
DOWNLOAD_DIR = "kaggle_data"
EXTRACT_DIR = "kaggle_data/extracted"


def download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.system(f'kaggle datasets download -d {DATASET} -p {DOWNLOAD_DIR}')


def extract():
    for f in os.listdir(DOWNLOAD_DIR):
        if f.endswith(".zip"):
            zip_path = os.path.join(DOWNLOAD_DIR, f)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(EXTRACT_DIR)


def preview():
    csv_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".csv")]
    if not csv_files:
        return
    first_csv = os.path.join(EXTRACT_DIR, csv_files[0])
    df = pd.read_csv(first_csv)
    print(df.head())


if __name__ == "__main__":
    download()
    extract()
    preview()
