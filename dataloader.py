""" Written by Youval Vanlaer
13/03/2021 """

import numpy as np
import os
import torch
import math
from torch.utils.data import Dataset, DataLoader

window_size = 7
min_note = 1
max_note = 88

def load_preprocess(folder):
    ipath = os.path.join(folder,'input.dat')
    note_range = max_note - min_note + 1
    n_bins = 7 * 36
    mmi = np.memmap(ipath, dtype="float32", mode='r')
    i = np.reshape(mmi,(-1,window_size,n_bins))
    opath = os.path.join(folder,'output.dat')
    mmo = np.memmap(opath, mode='r')
    o = np.reshape(mmo,(-1,note_range))
    return i,o

class MAPSDataset(Dataset):

    def __init__(self, inputs, outputs):
        # convert data to tensor
        self.inputs = torch.Tensor(inputs)
        self.outputs = torch.Tensor(outputs)

    def __getitem__(self, index):
        return (self.inputs[index], self.outputs[index])

    def __len__(self):
        return len(self.inputs)


class MAPSDataLoader(DataLoader):

    def __init__(self, folder):
        input,output = load_preprocess(folder)
        self.dataset = MAPSDataset(input,output)

        super().__init__(self.dataset)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)