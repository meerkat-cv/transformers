import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import math

class MixedDataset(Dataset):
    def __init__(self, datasets:list):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            else:
                index -= len(d)
        raise Exception("Index %d is not valid for dataset with length %d" % (index, len(self)))

class MixedSampler(Sampler):
    def __init__(self, data_source, proportion:list):
        self.data_source = data_source
        self.proportion = proportion
        print("Data source length: %d"  % (len(self.data_source),))
        print("Data source # datasets: %d"  % (len(self.data_source.datasets),))
        print("Data proportion: %s" % (self.proportion,))

        assert len(self.proportion) >= len(self.data_source.datasets)
        self.reset = [False] * len(self.proportion)

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        def _init_iterator_(x):
            return iter((torch.randperm(len(self.data_source.datasets[x]))[:4] + (0 if x==0 else len(self.data_source.datasets[x-1]))).tolist())

        its = []
        for i in range(len(self.data_source.datasets)):
            its.append(_init_iterator_(i))
        
        yielded_samples = 0
        while True:
            for i in range(len(its)):
                for _ in range(self.proportion[i]):
                    try:
                        yield next(its[i])
                        yielded_samples += 1
                    except StopIteration:
                        self.reset[i] = True
                        its[i] = _init_iterator_(i)
                        yield next(its[i])
                        yielded_samples += 1
                    finally:
                        if yielded_samples >= self.num_samples:
                            return

    def __len__(self):
        return self.num_samples

