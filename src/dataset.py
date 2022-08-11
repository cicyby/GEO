import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


############################################################################################
# This file provides basic processing script for the three signal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class MultiSignalDatasets(Dataset):
    def __init__(self, dataset_path, data='simulation', split_type='train', if_noise=False, label="vp"):
        super(MultiSignalDatasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data + '_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        if if_noise:
            self.rdispph = torch.tensor(dataset[split_type]['with_noise']['rdispph'], dtype=torch.float32)
            self.prf = torch.tensor(dataset[split_type]['with_noise']['prf'], dtype=torch.float32)
            self.rwe = torch.tensor(dataset[split_type]['with_noise']['rwe'], dtype=torch.float32)
        else:
            self.rdispph = torch.tensor(dataset[split_type]['without_noise']['rdispph'], dtype=torch.float32)
            self.prf = torch.tensor(dataset[split_type]['without_noise']['prf'], dtype=torch.float32)
            self.rwe = torch.tensor(dataset[split_type]['without_noise']['rwe'], dtype=torch.float32)
        self.labels = torch.tensor(dataset[split_type]['labels'][label], dtype=torch.float32)

        self.data = data
        self.n_modalities = 3

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.rdispph[1], self.prf.shape[1], self.rwe.shape[1]

    def get_dim(self):
        return self.text.rdispph[2], self.prf.shape[2], self.rwe.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = [self.rdispph[index], self.prf[index], self.rwe[index]]
        Y = self.labels[index]
        return X, Y

