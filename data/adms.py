import torch.utils.data as data
import os
import torch
import numpy as np
import random


def load_adms(root):
    path = os.path.join(root, 'aqms_after_IDW.pt')
    aqms = torch.load(path).float()[:, :, :]
    aqms = aqms.permute(2, 0, 1)
    path = os.path.join(root, 'diff_12.pt')
    adms = torch.load(path).float()[:, :, 20:]
    adms = adms.permute(2, 0, 1)
    path = os.path.join(root, 'wrf_after_cor.pt')
    wrf = torch.load(path)[:-20, ...]
    wrf = torch.from_numpy(wrf)

    # adms_max = torch.max(adms)
    # adms_min = torch.min(adms)
    # adms = (adms - adms_min) / (adms_max - adms_min)

    # adms = np.array(adms)
    # adms = np.cbrt(adms)
    # adms = torch.from_numpy(adms)

    # adms_std = torch.std(adms, False)  #20.0351
    # adms_mean = torch.mean(adms)  #-12.4155
    adms_std = 20.0351
    adms_mean = -12.4155
    adms = (adms - adms_mean) / adms_std

    wrf_std = torch.std(wrf, False)  #0.6242
    wrf_mean = torch.mean(wrf)  #0.3027
    wrf = (wrf - wrf_mean) / wrf_std
    # wrf_max = torch.max(wrf).
    # wrf_min = torch.min(wrf)
    # adms_max = torch.max(adms)
    # adms_min = torch.min(adms)

    # adms = torch.complex(adms, torch.tensor(0.))
    # adms = torch.pow(adms, 1/3)
    # adms = adms.real
    adms = np.array(adms)
    adms = np.cbrt(adms)
    adms = torch.from_numpy(adms)
    adms_min = torch.min(adms) # -2.2625
    adms_max = torch.max(adms) # 2.4410
    adms = (adms - adms_min) / (adms_max - adms_min)


    wrf = np.array(wrf)
    wrf = np.cbrt(wrf)
    wrf = torch.from_numpy(wrf)
    wrf_min = torch.min(wrf)  # -2.2625
    wrf_max = torch.max(wrf)  # 2.4410
    wrf = (wrf - wrf_min) / (wrf_max - wrf_min)

    return adms, aqms, wrf


class ADMS(data.Dataset):
    def __init__(self, root, is_train, mode):
        super(ADMS, self).__init__()
        self.adms, self.aqms, self.wrf = load_adms(root)
        # self.adms = self.adms.view(-1, 1, 240, 305)[:, :, :, :304]
        self.adms = self.adms.view(-1, 1, 240, 304)
        self.aqms = self.aqms.view(-1, 1, 240, 304)
        self.aqms = self.aqms[:, :, ::2, ::2]
        self.adms = self.adms[:, :, ::2, ::2]

        self.length = len(self.adms) - 72 - 24
        self.example_indices = list(range(self.length))

        # keep the same shuffle result, train:valid:test = 8:1:1
        r = random.random
        random.seed(2)
        # if mode != 'all':
        #     random.shuffle(self.example_indices, random=r)
        print(self.example_indices[:20])
        self.mode = mode
        if self.mode == 'train':
            self.length = 8 * (self.length // 10)
            self.example_indices = self.example_indices[:self.length]
            random.shuffle(self.example_indices, random=r)
        elif self.mode == 'valid':
            self.length = self.length // 10
            self.example_indices = self.example_indices[8*self.length:9*self.length]
        elif self.mode == 'test':
            self.length = self.length - 9 * (self.length // 10)
            self.example_indices = self.example_indices[-self.length:]
        self.is_train = is_train
        self.image_size_ = [240, 304]
        print(self.example_indices[:20])

    def __getitem__(self, idx):
        idx2 = self.example_indices[idx] + 72
        # print(idx2)
        input = self.adms[idx2-72:idx2, ...]
        # output = self.adms[idx2-1:idx2+23, ...]
        output = self.adms[idx2 + 23, ...]
        input_decoder = self.aqms[idx2-72:idx2, ...]
        wrf = self.wrf[idx2 - 1, ...].view(1, 6, 60, 76)
        # input_decoder = None
        out = [idx, output, input, input_decoder, wrf]
        return out

    def __len__(self):
        return self.length


if __name__ == "__main__":
    root = "data/"
    adms = ADMS(root, True, 'train')
