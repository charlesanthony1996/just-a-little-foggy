import os
import json
import pandas as pd
from collections import defaultdict



data_dir = "./data/defog"
meta_path = "./data/defog_metadata.csv"
output_json = "./data/defog_loso_splits.json"

def load_metadata(meta_path):
    meta = pd.read_csv(meta_path)
    meta["Filename"] = meta["Id"].astype(str) + ".csv"
    return meta


def build_subject_groups(meta, data_dir):
    all_files = set(os.listdir(data_dir))
    subjects = defaultdict(list)

    for _, row in meta.iterrows():
        fname = row["Filename"]
        subj = row["Subject"]

        if fname in all_files:
            subjects[subj].append(fname)

    return {s:fl for s, fl in subjects.items() if len(fl) > 0}


def generate_loso(subject_files):
    loso = {}
    subjects = list(subject_files.keys())

    for test_sub in subjects:
        test_files = subject_files[test_sub]
        train_files = []
        for s in subjects:
            if s != test_sub:
                train_files += subject_files[s]
        loso[test_sub] = {
            "train": sorted(train_files),
            "test": sorted(test_files)
        }
    
    return loso




def main():
    meta = load_metadata(meta_path)
    subject_files = build_subject_groups(meta, data_dir)
    loso = generate_loso(subject_files)

    with open(output_json, "w") as f:
        json.dump(loso, f, indent=4)

    print("loso splits saved: ", output_json)
    print("subjects: ", len(subject_files))

    with open("../data/defog_loso_splits.json", "r") as f:
        loso = json.load(f)

    rows = []
    for subject, splits in loso.items():
        num_files = len(splits["test"])
        rows.append((subject, num_files))

    # store it to a dataframe
    df = pd.DataFrame(rows, columns=["Subject", "NumFiles"])

    # save it to a csv file
    df.to_csv("defog_subject_session_counts.csv", index=False)

    


if __name__ == "__main__":
    main()

