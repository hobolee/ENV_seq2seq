import torchvision.transforms as transforms
import numpy as np
import torch
import os


transforms_GY = transforms.ToTensor()
transforms_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

class GY(object):
    def __init__(self, min, max):
        self.min_value = min
        self.max_value = max

    # transform 会调用该方法
    def __call__(self, img):
        img = (img - self.min_value) / (self.max_value - self.min_value)
        return img

class BZ(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # transform 会调用该方法
    def __call__(self, img):
        img = (img - self.mean) / self.std
        return img





root = r'C:\Users\lihaobo\Downloads\data\data_no2'
path = os.path.join(root, 'diff.pt')
adms = torch.load(path).float()
adms = adms.numpy()
# mean = np.mean(adms)
# std = np.std(adms)
max = np.max(adms) #242.62038
min = np.min(adms) #-281.55322
transform_compose = transforms.Compose([
    GY(min, max),
    # BZ(mean, std)
    # transforms_BZ
])

# for i in range(26304):
#     data = adms[:, :, i]
#     print(i)
#     if i == 0:
#         data_transform = transform_compose(data).reshape((240, 304, 1))
#     else:
#         data_transform = np.concatenate((data_transform, transform_compose(data).reshape((240, 304, 1))), axis=2)

# mean = np.mean(data_transform)
# std = np.std(data_transform)
# transform_compose = transforms.Compose([
#     BZ(mean, 0.5)
# ])
# for i in range(100):
#     data = adms[:, :, i]
#     if i == 0:
#         data_bz = transform_compose(data).reshape((240, 304, 1))
#     else:
#         data_bz = np.concatenate((data_bz, transform_compose(data).reshape((240, 304, 1))), axis=2)
#

data_transform = transform_compose(adms)
print(np.max(data_transform))
print(np.min(data_transform))
data = torch.from_numpy(data_transform)
torch.save(data, 'diff_after_norm.pt')
pass
