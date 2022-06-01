import torch.utils.data as data
import os
import torch
import numpy as np


def load_adms(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'data_adms.pt')
    adms = torch.load(path).float()[:200, :]
    return adms


def load_adms_fixed(root):
    # Load the fixed dataset
    path = os.path.join(root, 'data_adms.pt')
    dataset = torch.load(path).float()[:200, :]
    # dataset = torch.cat((dataset[:5381, :], dataset[5381, :].view(1, -1), dataset[5381:, :]))
    return dataset


class ADMS(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output):
        super(ADMS, self).__init__()

        self.dataset = None
        if is_train:
            self.adms = load_adms(root)
            self.adms = self.adms.view(-1, 1, 240, 305)[:, :, :, :304]
        else:
            self.dataset = load_adms_fixed(root)
            self.dataset = self.dataset.view(-1, 1, 240, 305)[:, :, :, :304]
            pass # to do
        # self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]
        if not is_train:
            self.length = 10
        else:
            self.length = self.adms.shape[0] - 72 - 24
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.image_size_ = [240, 304]
        # self.digit_size_ = 28
        # self.step_length_ = 0.1

    def __getitem__(self, idx):
        idx2 = idx+72
        if self.is_train:
            input = self.adms[idx2-72:idx2, :, :, :]
            output = self.adms[idx2+24, :, :, :]
        else:
            input = self.dataset[idx2 - 72:idx2, :, :, :]
            output = self.dataset[idx2 + 24, :, :, :]
        out = [idx, output, input]
        return out

    def __len__(self):
        return self.length

        pass


if __name__ == "__main__":
    root = "data/"
    adms = ADMS(root, True, 10, 10)
