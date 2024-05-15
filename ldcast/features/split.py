from bisect import bisect_left
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from . import batch


def get_chunks(
    primary_raw, valid_frac=0.1, test_frac=0.1,
    chunk_seconds=2*24*60*60, random_seed=None
):
    t0 = min(
        primary_raw["patch_times"][0],
        primary_raw["zero_patch_times"][0]
    )
    t1 = max(
        primary_raw["patch_times"][-1],
        primary_raw["zero_patch_times"][-1]
    )+1

    rng = np.random.RandomState(seed=random_seed)
    chunk_limits = np.arange(t0,t1,chunk_seconds)
    num_chunks = len(chunk_limits)-1
    
    chunk_ind = np.arange(num_chunks)
    rng.shuffle(chunk_ind)
    i_valid = int(round(num_chunks * valid_frac))
    i_test = i_valid + int(round(num_chunks * test_frac))
    chunk_ind = {
        "valid": chunk_ind[:i_valid],
        "test": chunk_ind[i_valid:i_test],
        "train": chunk_ind[i_test:]
    }
    def get_chunk_limits(chunk_ind_split):
        return sorted(
            (chunk_limits[i], chunk_limits[i+1])
            for i in chunk_ind_split
        )
    chunks = {
        split: get_chunk_limits(chunk_ind_split)
        for (split, chunk_ind_split) in chunk_ind.items()
    }
    return chunks


def train_valid_test_split(
    raw_data, primary_raw_var, chunks=None, **kwargs
):
    if chunks is None:
        primary = raw_data[primary_raw_var] 
        chunks = get_chunks(primary, **kwargs)

    def split_chunks_from_array(x, chunks_split, times):
        n = 0
        chunk_ind = []
        for (t0,t1) in chunks_split:
            k0 = bisect_left(times, t0)
            k1 = bisect_left(times, t1)
            n += k1 - k0
            chunk_ind.append((k0,k1))
        
        shape = (n,) + x.shape[1:]
        x_chunk = np.empty_like(x, shape=shape)
        
        j0 = 0
        for (k0,k1) in chunk_ind:
            j1 = j0 + (k1-k0)
            x_chunk[j0:j1,...] = x[k0:k1,...]
            j0 = j1

        return x_chunk

    split_raw_data = {
        split: {var: {} for var in raw_data}
        for split in chunks
    }
    for (var, raw_data_var) in raw_data.items():
        for (split, chunks_split) in chunks.items():
            
            split_raw_data[split][var]["patches"] = \
                split_chunks_from_array(
                    raw_data_var["patches"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["patch_coords"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["patch_times"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["zero_patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_coords"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )
            split_raw_data[split][var]["zero_patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_times"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )

            added_keys = set(split_raw_data[split][var].keys())
            missing_keys = set(raw_data[var].keys()) - added_keys
            for k in missing_keys:
                split_raw_data[split][var][k] = raw_data[var][k]

    return (split_raw_data, chunks)


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        variables, raw, predictors, target, primary_var,
        sampling_bins, sampler_file,
        batch_size=64,
        train_epoch_size=1000, valid_epoch_size=200, test_epoch_size=1000,
        valid_seed=None, test_seed=None,
        **kwargs
    ):
        super().__init__()
        self.batch_gen = {
            split: batch.BatchGenerator(
                variables, raw_var, predictors, target, primary_var,
                sampling_bins=sampling_bins, batch_size=batch_size,
                sampler_file=sampler_file.get(split),
                augment=(split=="train"),
                **kwargs
            )
            for (split,raw_var) in raw.items()
        }
        self.datasets = {}
        if "train" in self.batch_gen:
            self.datasets["train"] = batch.StreamBatchDataset(
                self.batch_gen["train"], train_epoch_size
            )
        if "valid" in self.batch_gen:
            self.datasets["valid"] = batch.DeterministicBatchDataset(
                self.batch_gen["valid"], valid_epoch_size, random_seed=valid_seed
            )
        if "test" in self.batch_gen:
             self.datasets["test"] = batch.DeterministicBatchDataset(
                self.batch_gen["test"], test_epoch_size, random_seed=test_seed
            )

    def dataloader(self, split):
        return DataLoader(
            self.datasets[split], batch_size=None,
            pin_memory=True, num_workers=0
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("valid")

    def test_dataloader(self):
        return self.dataloader("test")

class KNMIDataset(Dataset):
    """Custom dataset for loading preprocessed KNMI data

    Args:
        list_IDs (list): List of tuples, each containing sequences of input and target filenames.
        data_path (str): Path to the directory containing the preprocessed data files.
        x_seq_size (int): Number of timesteps in the input sequence.
        y_seq_size (int): Number of timesteps in the target sequence.
        load_prep (bool): Flag indicating whether the data is preprocessed.
    """

    def __init__(self, list_IDs, data_path, x_seq_size=6, y_seq_size=3, load_prep=True):
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size
        self.load_prep = load_prep

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.x_seq_size, 256, 256, 1), dtype=np.float32)
        y = np.empty((self.y_seq_size, 256, 256, 1), dtype=np.float32)
        
        x_IDs, y_IDs = list_IDs_temp
        for t in range(self.x_seq_size):
            X[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(x_IDs[t])))

        for t in range(self.y_seq_size):
            y[t] = np.load(os.path.join(self.data_path, '{}.npy'.format(y_IDs[t])))

        # Permute to match the required shape: (x_seq_size, img_width, img_height, 1) to (1, 1, x_seq_size, img_width, img_height)
        X = torch.tensor(X).permute(3, 0, 1, 2).unsqueeze(0)
        y = torch.tensor(y).permute(3, 0, 1, 2).unsqueeze(0)

        return X, y

# Custom collate function for batching the data
# This ensures that each batch of KNMI data is in the right shape for the model
def collate_fn(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    data = torch.stack(data)  # (batch_size, 1, 1, X_sequence_length, 256, 256)
    targets = torch.stack(targets)  # (batch_size, 1, Y_sequence_length, 256, 256)

    # Create timestamps tensor for the batch
    batch_size = data.size(0)
    x_seq_size = data.size(2)
    timestamps = torch.arange(x_seq_size).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, X_sequence_length)

    return [[data, timestamps]], targets

class KNMIDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for loading KNMI data

    Args:
        train_data (list): List of training data file IDs.
        val_data (list): List of validation data file IDs.
        data_path (str): Path to the directory containing the preprocessed data files.
        batch_size (int): Batch size for the DataLoader.
        x_seq_size (int): Number of timesteps in the input sequence.
        y_seq_size (int): Number of timesteps in the target sequence.
    """
    def __init__(self, train_data, val_data, data_path, batch_size=32, x_seq_size=6, y_seq_size=3):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.data_path = data_path
        self.batch_size = batch_size
        self.x_seq_size = x_seq_size
        self.y_seq_size = y_seq_size

    def setup(self, stage=None):
        self.train_dataset = KNMIDataset(self.train_data, self.data_path, 
                                           x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size)
        self.val_dataset = KNMIDataset(self.val_data, self.data_path, 
                                         x_seq_size=self.x_seq_size, y_seq_size=self.y_seq_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

# Usage example:
# train_IDs = np.load('datasets/train_randomsplit.npy', allow_pickle=True)
# val_IDs = np.load('datasets/val_randomsplit.npy', allow_pickle=True)
# data_path = '/restore/knmimo/preprocessed/rtcor/'

# data_module = CustomDataModule(train_data=train_IDs, val_data=val_IDs, data_path=data_path, batch_size=32)
# data_module.setup()
# train_loader = data_module.train_dataloader()
# val_loader = data_module.val_dataloader()
