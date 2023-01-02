import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from typing import Tuple, Sequence, Any, List
from torch.utils.data import WeightedRandomSampler, DataLoader, BatchSampler
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split


def get_df_from_csv(dataset_csv_path: str, dataset_dir: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv_path, header=None, names=["path", "age"])
    df["path"] = dataset_dir + df["path"]
    return df

def get_df_with_age(df: pd.DataFrame, age: int) -> pd.DataFrame:
    return df.loc[df['age'] == age]

def get_df_within_range(df: pd.DataFrame, label_col: str, range_min: int, range_max: int) -> pd.DataFrame:
    return df.loc[df[label_col].between(range_min, range_max)]

def get_df_with_age_subdivision(df: pd.DataFrame, n_subdivisions: int) -> Tuple[pd.DataFrame, dict]:
    def _get_df_with_age_subdivision(df: pd.DataFrame, strict_limit):
        subdivisions = {}

        sub_size = int(len(df) / n_subdivisions)
        age_size = df["age"].value_counts().sort_index()
        current_sub = 0
        current_sub_size = 0
        
        for i in age_size.index:
            if not strict_limit:
                if current_sub_size + age_size.loc[i] > sub_size:
                    current_sub += 1
                    current_sub_size = 0

            subdivisions[i] = current_sub
            current_sub_size += age_size.loc[i]

            if strict_limit:
                if current_sub_size > sub_size:
                    current_sub += 1
                    current_sub_size = 0
        df["age"] = df["age"].apply(lambda x: subdivisions[x])
        real_subdivisions = len(df["age"].value_counts().index)
        return df, real_subdivisions, subdivisions
    _df, real_subdivisions, subdivisions = _get_df_with_age_subdivision(df.copy(), True)
    if real_subdivisions != n_subdivisions:
        _df, real_subdivisions, subdivisions = _get_df_with_age_subdivision(df.copy(), False)
    return _df, subdivisions

def get_starting_class(df: pd.DataFrame, label_col: str) -> int:
    return df[label_col].value_counts().sort_index().index.min()

def get_n_classes(df: pd.DataFrame, label_col: str) -> int:
    return df[label_col].value_counts().sort_index().index.max() + 1 - get_starting_class(df, label_col)

def expected_to_value(y: torch.Tensor):
    return torch.sum(torch.arange(y.shape[-1]) * y, axis=-1)

def get_label_map_vector():
    def _to_kl_labels(y, n_classes):
        std = 1.0
        _y = np.arange(n_classes)
        return 1/(std * np.sqrt(2*np.pi)) * np.exp(-np.square(_y-y) / (2*std**2))

    label_map_v = {x: np.zeros(shape=(8)) for x in range(1, 82)}

    for lm in label_map_v:
        kl = _to_kl_labels(lm, 81)
        kl_class = np.zeros(shape=(8))
        for i, value in enumerate(kl):
            idx = 7 if int((i-1)/10) > 7 else int((i-1)/10)
            kl_class[idx] += value
        label_map_v[lm] = kl_class
    return label_map_v