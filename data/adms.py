import torch.utils.data as data
import os
import torch
import numpy as np
import random


def load_adms(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'data_adms.pt')
    adms = torch.load(path).float()[:, :]
    # adms = adms.permute(2, 0, 1)
    return adms


def load_adms_fixed(root):
    # Load the fixed dataset
    path = os.path.join(root, 'data_adms.pt')
    dataset = torch.load(path).float()[:200, :]
    # dataset = torch.cat((dataset[:5381, :], dataset[5381, :].view(1, -1), dataset[5381:, :]))
    return dataset


class ADMS(data.Dataset):
    def __init__(self, root, is_train, mode):
        super(ADMS, self).__init__()
        # self.dataset = None
        # if is_train:
        #     self.adms = load_adms(root)
        #     self.adms = self.adms.view(-1, 1, 240, 305)[:, :, :, :304]
        # else:
        #     self.dataset = load_adms_fixed(root)
        #     self.dataset = self.dataset.view(-1, 1, 240, 305)[:, :, :, :304]
        #     pass # to do
        # self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]

        self.adms = load_adms(root)
        self.adms = self.adms.view(-1, 1, 240, 305)[:, :, :, :304]
        self.adms = self.adms[:, :, ::2, ::2]
        # self.adms = self.adms.view(-1, 1, 240, 304)
        self.length = len(self.adms) - 48 - 24
        self.example_indices = list(range(self.length))

        # keep the same shuffle result, train:valid:test = 8:1:1
        r = random.random
        random.seed(2)
        if mode != 'all':
            random.shuffle(self.example_indices, random=r)
        print(self.example_indices[:20])
        self.mode = mode
        if self.mode == 'train':
            self.length = 8 * self.length // 10
            self.example_indices = self.example_indices[:self.length]
        elif self.mode == 'valid':
            self.length = self.length // 10
            self.example_indices = self.example_indices[8*self.length:9*self.length]
        elif self.mode == 'test':
            self.length = self.length - 9 * self.length // 10
            self.example_indices = self.example_indices[-self.length:]
        self.is_train = is_train
        self.image_size_ = [240, 304]

    def __getitem__(self, idx):
        idx2 = self.example_indices[idx] + 48
        print(idx2)
        input = self.adms[idx2-48:idx2, :, :, :]
        output = self.adms[idx2+23, :, :, :]
        out = [idx, output, input]
        return out

    def __len__(self):
        return self.length


if __name__ == "__main__":
    root = "data/"
    adms = ADMS(root, True, 'train')
