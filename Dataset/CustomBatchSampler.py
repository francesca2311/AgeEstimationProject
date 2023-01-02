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
from Dataset import CustomDataset


class BalancedBatchSampler(BatchSampler):
    def __init__(self, img_labels: pd.Series, class_ranges: Sequence[Tuple[int, int]], samples_per_class: int) -> None:
        self.img_labels = img_labels
        self.class_ranges = class_ranges
        self.samples_per_class = samples_per_class
        self.batch_size = len(class_ranges) * samples_per_class
        self.n_batches = int(len(img_labels) / self.batch_size)
        self.indices_per_class_range = self._get_paths_per_class_range()

    def _get_paths_per_class_range(self) -> dict:
        indices_per_class_range = {x: [] for x in range(len(self.class_ranges))}
        for i, img_label in enumerate(self.img_labels):
            for class_range_index, class_range in enumerate(self.class_ranges):
                if img_label in range(*class_range):
                    indices_per_class_range[class_range_index] += [i]
        return indices_per_class_range

    def __iter__(self) -> List:
        for _ in range(self.n_batches):
            batch = []
            for class_range in self.indices_per_class_range:
                batch += random.choices(self.indices_per_class_range[class_range], k=self.samples_per_class)
            yield batch
        
    def __len__(self) -> int:
        return self.n_batches


class RandomlyBalancedBatchSampler(BalancedBatchSampler):
    def __init__(self, img_labels: pd.Series, class_ranges: Sequence[Tuple[int, int]], samples_per_class: int, balanced_prob: float) -> None:
        super().__init__(img_labels, class_ranges, samples_per_class)
        self.balanced_prob = balanced_prob

    def __iter__(self) -> List:
        for _ in range(self.n_batches):
            if random.random() < self.balanced_prob:
                batch = []
                for class_range in self.indices_per_class_range:
                    batch += random.choices(self.indices_per_class_range[class_range], k=self.samples_per_class)
                yield batch
            else:
                yield random.choices(list(range(len(self.img_labels))), k=self.samples_per_class * len(self.class_ranges))


class RandomClassBatchSampler(BatchSampler):
    def __init__(self, img_labels: pd.Series, class_ranges: Sequence[Tuple[int, int]], batch_size: int) -> None:
        self.img_labels = img_labels
        self.class_ranges = class_ranges
        self.batch_size = batch_size
        self.n_batches = int(len(img_labels) / self.batch_size)
        self.indices_per_class_range = self._get_paths_per_class_range()

    def _get_paths_per_class_range(self) -> dict:
        indices_per_class_range = {x: [] for x in range(len(self.class_ranges))}
        for i, img_label in enumerate(self.img_labels):
            for class_range_index, class_range in enumerate(self.class_ranges):
                if img_label in range(*class_range):
                    indices_per_class_range[class_range_index] += [i]
        return indices_per_class_range

    def __iter__(self) -> List:
        n_classes = len(self.class_ranges)
        for _ in range(self.n_batches):
            batch = []
            for i in range(self.batch_size):
                r = random.randrange(0, n_classes)
                batch += random.choices(self.indices_per_class_range[r], k=1)
            yield batch
        
    def __len__(self) -> int:
        return self.n_batches