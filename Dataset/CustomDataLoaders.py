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
from Dataset.CustomBatchSampler import BalancedBatchSampler


class CustomDataLoader:
    def __init__(self, dataset: StandardDataset) -> None:
        self._dataset: StandardDataset = dataset

    def get_balanced_dataloader(self, class_ranges, samples_per_class, **kwargs: Any) -> DataLoader:
        sampler = BalancedBatchSampler(self._dataset.get_img_labels(), class_ranges, samples_per_class)
        return DataLoader(self._dataset, batch_sampler=sampler, **kwargs)

    def get_unbalanced_dataloader(self, batch_size, shuffle=True, **kwargs: Any) -> DataLoader:
        return DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)