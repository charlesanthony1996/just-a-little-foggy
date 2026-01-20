import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

tdcsfog_dataset = "charlesanthony1996/mjff-train-tdcsfog-folder"
defog_dataset = "charlesanthony1996/mjff-train-defog-folder"
notype_dataset = "charlesanthony1996/mjff-train-notype-folder"

this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, "data")

def download_and_extract(dataset_id: str, subdir: str):
    out_dir = os.path.join(data_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    tmp_zip = os.path.join(out_dir)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset_id} into {tmp_zip} ...")
    api.dataset_download_files(dataset_id, path=out_dir, quiet=False)

    zip_files = [f for f in os.listdir(out_dir) if f.endswith(".zip")]
    if not zip_files:
        return
    
    zip_path = os.path.join(out_dir, zip_files[0])
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


    os.remove(zip_path)
    print(f"finished dataset id: {out_dir}")



def download_all():
    os.makedirs(data_dir, exist_ok=True)

    download_and_extract(tdcsfog_dataset, "tdcsfog")
    download_and_extract(defog_dataset, "defog")
    download_and_extract(notype_dataset, "notype")

if __name__ == "__main__":
    download_all()