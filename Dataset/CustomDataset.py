import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from typing import Tuple, Sequence, Any, List, Dict
from torch.utils.data import WeightedRandomSampler, DataLoader, BatchSampler
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
from Utils import CSVUtils


class StandardDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str, label_col: str, transform_func=None, label_function: str="CAE") -> None:
        self._img_paths = df[path_col]
        self._img_labels = df[label_col]
        self._label_function = self._get_label_function(label_function)
        self._transform_func = (lambda x: np.array(x)) if transform_func is None else transform_func
        self._starting_class = CSVUtils.get_starting_class(df, label_col)
        self._n_classes = CSVUtils.get_n_classes(df, label_col)

    def get_img_paths(self) -> pd.Series:
        return self._img_paths

    def get_img_labels(self) -> pd.Series:
        return self._img_labels

    def set_starting_class(self, starting_class) -> None:
        self._starting_class = starting_class

    def set_n_classes(self, n_classes) -> None:
        self._n_classes = n_classes

    def _to_categorical(self, y, n_classes) -> np.ndarray:
        return np.eye(n_classes)[y]

    def _to_soft_labels(self, y, n_classes) -> np.ndarray:
        """From the paper "Effective training of convolutional networks for face-based gender and age prediction" by Antipov et al.
        std with 2.5 as default as proposed in the paper."""
        std = 2.5
        _y = np.arange(n_classes)
        return 1/(std * np.sqrt(2*np.pi)) * np.exp(-np.square(_y-y) / (2*std**2))

    def _get_label_function(self, label_function):
        if label_function == "CAE":
            return self._to_categorical
        if label_function == "LDAE":
            return self._to_soft_labels
        return lambda x: np.array(x)

    def _get_image(self, image_path):
        return self._transform_func(Image.open(image_path))

    def _get_label(self, label):
        return self._label_function(label-self._starting_class, self._n_classes)

    def __len__(self) -> int:
        return len(self._img_labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_image(self._img_paths[idx]), self._get_label(self._img_labels[idx])


class AgeGroupAndAgeDataset(StandardDataset):
    def __init__(self, df: pd.DataFrame, path_col: str, label_col: str,
                 label_map: Dict, label_map_n_classes: int, transform_func=None, label_function: str="CAE") -> None:
        super().__init__(df, path_col, label_col, transform_func, label_function)
        self.label_map = label_map
        self.label_map_n_classes = label_map_n_classes

    def _get_label(self, label):
        return self._to_categorical(self.label_map[label], self.label_map_n_classes), self._label_function(label-self._starting_class, self._n_classes)