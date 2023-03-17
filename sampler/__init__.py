# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import os
from .utils import read_texts
from .DataLoader import TranslationDataset, DataLoader, RewardDataset

data_path = './outputs/data.txt'


class ppo_sampler:
    def __init__(self, batch_size):
        self.iter = DataLoader(TranslationDataset(), batch_size=batch_size, shuffle=True)

    def iterator(self):
        return self.iter


class rm_sampler:
    def __init__(self, sft, num_sample):
        self.iter = read_texts(data_path) if os.path.exists(data_path) else RewardDataset(sft, num_sample).input_texts

    def iterator(self):
        return self.iter
