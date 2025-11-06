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
    print("\n📥 Downloading dataset...\n")
    os.system(f'kaggle datasets download -d {DATASET} -p {DOWNLOAD_DIR}')
    print("✅ Download complete!")


def extract():
    for f in os.listdir(DOWNLOAD_DIR):
        if f.endswith(".zip"):
            zip_path = os.path.join(DOWNLOAD_DIR, f)
            print(f"\n📦 Extracting {zip_path}...\n")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(EXTRACT_DIR)
            print("✅ Extracted!")


def preview():
    csv_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found after extraction.")
        return
    first_csv = os.path.join(EXTRACT_DIR, csv_files[0])
    print(f"\n📄 Previewing first file: {first_csv}\n")
    df = pd.read_csv(first_csv)
    print(df.head())
    print("\n🧠 Columns:", list(df.columns))


if __name__ == "__main__":
    download()
    extract()
    preview()
