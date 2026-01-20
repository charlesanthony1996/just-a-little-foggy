import pandas as pd
from splits_kfold import build_group_kfold
from kfold_train import run_kfold
# from your_metadata_loader import build_subject_groups, load_metadata
from defog_loso import build_subject_groups, load_metadata

DATA_DIR = "./data/defog"
META_PATH = "./data/defog_metadata.csv"

meta = load_metadata(META_PATH)
subject_files = build_subject_groups(meta, DATA_DIR)

df = run_kfold(
    subject_files,
    DATA_DIR,
    model_name="lstm",
    k=5,
    epochs=10
)

print("\n=== FINAL RESULTS ===")
print(df)

print("\nMEAN ± STD")
print(df.mean(numeric_only=True))
print(df.std(numeric_only=True))
