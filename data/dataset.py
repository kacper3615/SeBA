import os
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(P, dataset, batch_size, seed):
    path = os.path.join(os.path.dirname(__file__), f"datafiles/{dataset}.arff")
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("|S"):
            df[col] = df[col].astype(str)

    target_col = "class"
    y_series = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str).values)
    P.class_names = list(map(str, le.classes_))
    P.num_classes = len(P.class_names)

    X_temp_df, X_test_df, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp_df, y_temp, test_size=0.1, random_state=seed, stratify=y_temp
    )

    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=["object"]).columns.tolist()

    if num_cols:
        scaler = StandardScaler()
        X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols])
        X_val_df[num_cols] = scaler.transform(X_val_df[num_cols])
        X_test_df[num_cols] = scaler.transform(X_test_df[num_cols])

    X_train_df = pd.get_dummies(X_train_df, columns=cat_cols)
    X_val_df = pd.get_dummies(X_val_df, columns=cat_cols)
    X_test_df = pd.get_dummies(X_test_df, columns=cat_cols)

    X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0)

    feature_groups, curr = {}, 0
    for col in X.columns:
        if col in num_cols:
            feature_groups[col] = [curr]
            curr += 1
        else:
            ohe_cols = [c for c in X_train_df.columns if c.startswith(col + "_")]
            feature_groups[col] = list(range(curr, curr + len(ohe_cols)))
            curr += len(ohe_cols)

    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    P.input_dim = X_train.shape[1]

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, feature_groups


def sample_support_query(dataset, shot):
    if hasattr(dataset, "y"):
        labels = dataset.y.tolist()
    elif hasattr(dataset, "targets"):
        labels = (
            dataset.targets
            if isinstance(dataset.targets, list)
            else dataset.targets.tolist()
        )
    else:
        raise AttributeError("Dataset doesn't have 'y' or 'targets' attributes")

    classes = sorted(set(labels))
    class_to_idxs = {c: [] for c in classes}
    for idx, lbl in enumerate(labels):
        class_to_idxs[lbl].append(idx)

    support_idxs, query_idxs = [], []
    for c in classes:
        idxs = class_to_idxs[c]
        if len(idxs) < shot:
            raise ValueError(
                f"Class {c} has only {len(idxs)} examples; need >= {shot}."
            )
        sampled = random.sample(idxs, shot)
        support_idxs += sampled
        query_idxs += [i for i in idxs if i not in sampled]
    return support_idxs, query_idxs
