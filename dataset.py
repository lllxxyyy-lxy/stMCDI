import pickle
import yaml
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

def process_func(path: str, aug_rate=1, missing_ratio=0.1):
    data = pd.read_csv(path, header=None, sep='\t')
    # print(data.head())

    data = data.iloc[1:, :]
    data.replace("?", np.nan, inplace=True)
    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()
    # for each column, mask `missing_ratio` % of observed values.
    for col in range(observed_values.shape[1]):  # col #
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False
    # gt_mask: 0 for missing elements and manully maksed elements
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)

    # observed_masks: 0 for missing elements
    observed_masks = observed_masks.astype(int)  # "float32"
    gt_masks = gt_masks.astype("float32")
    if observed_values.shape[1] == 0:
        raise ValueError("The sequence length (L) is 0. Please check the input data.")

    return observed_values, observed_masks, gt_masks


class st_dataset(Dataset):
    def __init__(
        self, eval_length=1000, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0
    ):
        self.eval_length = eval_length
        np.random.seed(seed)

        dataset_path = "data/Dataset44/Insitu_count.txt"
        location_path = "data/Dataset44/Locations.txt"
        self.observed_values, self.observed_masks, self.gt_masks = process_func(
            dataset_path, aug_rate=aug_rate, missing_ratio=missing_ratio
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        # connected_edges = self.edge_index[:, self.edge_index[0] == index]
        if index >= len(self.observed_values):
            raise IndexError(f"Index {index} is out of bounds for edge_index with size {len(self.edge_index)}")

        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            # "edge_index": connected_edges
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
    dataset = st_dataset(missing_ratio=missing_ratio, seed=seed)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    # Here we perform max-min normalization.
    processed_data_path_norm = (
        f"./data/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    )
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # Data transformation after train-test split.
        col_num = dataset.observed_values.shape[1]
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for k in range(col_num):
            # Using observed_mask to avoid counting missing values (now represented as 0)
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])
        # print(f"--------------Max-value for each column {max_arr}--------------")
        # print(f"--------------Min-value for each column {min_arr}--------------")

        dataset.observed_values = (
            (dataset.observed_values - (min_arr - 1)) / (max_arr - min_arr + 1)
        ) * dataset.observed_masks


    # Create datasets and corresponding data loaders objects.
    train_dataset = st_dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Training Dataset created--------")
    print(f"Training dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=0)

    valid_dataset = st_dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Validation Dataset created--------")
    print(f"Validation dataset size: {len(valid_dataset)}")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = st_dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Test Dataset created--------")
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)


    return train_loader, valid_loader, test_loader


