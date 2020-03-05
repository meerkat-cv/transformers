import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.mixed_sampling import MixedDataset, MixedSampler

data_A = TensorDataset(torch.tensor([0,1,2,3]))                 # pylint: disable=not-callable
data_B = TensorDataset(torch.tensor([10, 11]))              # pylint: disable=not-callable

data = MixedDataset([data_A, data_B])
bs = 4
sampler_torch = MixedSampler(data, [1, 3])

for i in DataLoader(data, sampler=sampler_torch, batch_size=bs):
    print(i)
