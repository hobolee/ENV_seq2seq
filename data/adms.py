import torch.utils.data as data
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def load_adms(root):
    path = os.path.join(root, 'aqms_after_IDW.pt')
    aqms = torch.load(path).float()[:, :, 20:]
    aqms = aqms.permute(2, 0, 1)
    path = os.path.join(root, 'adms_after_cor.pt')
    adms = torch.load(path).float()[:, :, 20:]
    adms = adms.permute(2, 0, 1)
    # path = os.path.join(root, 'diff_12.pt')
    # adms = torch.load(path).float()[:, :, 20:1000]
    # adms = adms.permute(2, 0, 1)

    # plt.figure()
    # plt.plot(adms[:, 120, 154], 'b')
    # plt.plot(aqms[:, 120, 154] * 1.88, 'r')

    path = os.path.join(root, 'wrf_after_cor.pt')
    wrf = torch.load(path)[:-20, ...]
    wrf = torch.from_numpy(wrf)

    aqms_std = torch.std(aqms, False)  #11.5286
    aqms_mean = torch.mean(aqms)  #21.9199
    aqms = (aqms - aqms_mean) / aqms_std

    adms_std = torch.std(adms, False)  # 22.8285
    adms_mean = torch.mean(adms)  # 23.8102
    # adms_std = 22.8285
    # adms_mean = 23.8102
    adms = (adms - adms_mean) / adms_std

    adms = np.array(adms)
    adms = np.cbrt(adms)
    adms = torch.from_numpy(adms)
    adms_min = torch.min(adms) # -1.7875
    adms_max = torch.max(adms) # 2.8531
    adms = (adms - adms_min) / (adms_max - adms_min)

    wrf_std = torch.std(wrf, False)  #0.6242
    wrf_mean = torch.mean(wrf)  #0.3027
    wrf = (wrf - wrf_mean) / wrf_std
    wrf = np.array(wrf)
    wrf = np.cbrt(wrf)
    wrf = torch.from_numpy(wrf)
    wrf_min = torch.min(wrf)  # -2.2625
    wrf_max = torch.max(wrf)  # 2.4410
    wrf = (wrf - wrf_min) / (wrf_max - wrf_min)

    # plt.figure()
    # adms = torch.pow(adms[:, 120, 154] * (2.8531 + 1.7875) - 1.7875, 3) * 22.8285 + 23.8102
    # plt.plot(adms, 'b')
    # plt.plot((aqms[:, 120, 154] * aqms_std + aqms_mean) * 1.88, 'r')
    # plt.show()

    return adms, aqms, wrf


class ADMS(data.Dataset):
    def __init__(self, root, is_train, mode):
        super(ADMS, self).__init__()
        self.adms, self.aqms, self.wrf = load_adms(root)
        self.adms = self.adms.view(-1, 1, 240, 304)#[:, :, :, :304]
        # self.adms = self.adms.view(-1, 1, 240, 304)
        self.aqms = self.aqms.view(-1, 1, 240, 304)
        self.aqms = self.aqms[:, :, ::1, ::1]
        self.adms = self.adms[:, :, ::1, ::1]

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
        input = self.aqms[idx2-72:idx2, ...]
        # output = self.adms[idx2-1:idx2+23, ...]
        output = self.adms[idx2 - 1, ...]
        input_decoder = self.aqms[idx2-72:idx2, ...]
        wrf = self.wrf[idx2-72:idx2, ...]#.view(1, 6, 60, 76)
        # input_decoder = None
        out = [idx, output, input, input_decoder, wrf]
        return out

    def __len__(self):
        return self.length


if __name__ == "__main__":
    root = "data/"
    adms = ADMS(root, True, 'train')
