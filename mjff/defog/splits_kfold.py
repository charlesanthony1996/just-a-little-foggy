import numpy as np
from sklearn.model_selection import KFold

def build_group_kfold(subject_files, k=5, seed=42):
    """
    subject_files: dict {subject_id: [file1.csv, file2.csv, ...]}
    """
    subjects = np.array(list(subject_files.keys()))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        train_subjects = subjects[train_idx]
        test_subjects = subjects[test_idx]

        train_files = []
        test_files = []

        for s in train_subjects:
            train_files.extend(subject_files[s])
        for s in test_subjects:
            test_files.extend(subject_files[s])

        folds.append({
            "fold": fold_idx,
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects),
            "train_files": sorted(train_files),
            "test_files": sorted(test_files),
        })

    return folds
