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
from Dataset.CustomDataset import StandardDataset
from Dataset.CustomBatchSampler import BalancedBatchSampler, RandomlyBalancedBatchSampler, RandomClassBatchSampler, RandomClassAndDatasetBatchSampler, RandomClassBatchSampler2, RandomAgeBatchSampler2


class CustomDataLoader:
    def __init__(self, dataset: StandardDataset) -> None:
        self._dataset: StandardDataset = dataset

    def get_balanced_dataloader(self, class_ranges, samples_per_class, **kwargs: Any) -> DataLoader:
        sampler = BalancedBatchSampler(self._dataset.get_img_labels(), class_ranges, samples_per_class)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs)

    def get_balanced_class_dataloader(self, class_ranges, batch_size, **kwargs: Any) -> DataLoader:
        sampler = RandomClassBatchSampler(self._dataset.get_img_labels(), class_ranges, batch_size)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs)

    def get_balanced_class_dataloader2(self, class_ranges, batch_size, **kwargs: Any) -> Tuple[DataLoader, RandomClassBatchSampler2]:
        sampler = RandomClassBatchSampler2(self._dataset.get_img_labels(), class_ranges, batch_size)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs), sampler

    def get_balanced_age_dataloader2(self, batch_size, **kwargs: Any) -> Tuple[DataLoader, RandomClassBatchSampler2]:
        sampler = RandomAgeBatchSampler2(self._dataset.get_img_labels(), batch_size)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs), sampler

    def get_balanced_class_random_dataset_dataloader(self, df_base_len, class_ranges, batch_size, aug_prob=1.0, **kwargs: Any) -> Tuple[DataLoader, RandomClassAndDatasetBatchSampler]:
        sampler = RandomClassAndDatasetBatchSampler(self._dataset.get_img_labels(), df_base_len, class_ranges, batch_size, aug_prob)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs), sampler

    def get_randomly_balanced_dataloader(self, class_ranges, samples_per_class, balanced_prob, **kwargs: Any) -> DataLoader:
        sampler = RandomlyBalancedBatchSampler(self._dataset.get_img_labels(), class_ranges, samples_per_class, balanced_prob)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs)

    def get_unbalanced_dataloader(self, batch_size, shuffle=True, **kwargs: Any) -> DataLoader:
        return DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)