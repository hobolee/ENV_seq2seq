import sys

import numpy as np
import pandas as pd
from lib import DataInterpolate
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import torch


dt = pd.read_csv("/Users/lihaobo/PycharmProjects/ENV/NO2/AQ_NO2-19800101-20220504.csv", header=4, dtype=str)
dt = dt.values
index = 0
DI = DataInterpolate(dt, index)
DI.generate_dataset(48, [2019, 2021])
# DI.period_factor(index=list(np.arange(14)))
# print(DI.operate_data_period.min(axis=0))
# a = DI.operate_data_period
# np.save('aqms_after_interpolation.npy', a, allow_pickle=True)
aqms = np.load('aqms_after_interpolation.npy', allow_pickle=True)


lng_lat = np.load('/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]

# cal weight
# weight = np.zeros([240, 304, 12])
# d = np.zeros([240, 304, 12])
# stations = DI.location
# stations = np.delete(stations, [0, 4], 1)
#
# for i in range(240):
#     print(i)
#     for j in range(304):
#         for k in range(12):
#             d[i, j, k] = np.linalg.norm(np.array((lat[i, j], lon[i, j])) - stations[:, k].astype(float))
#         for k in range(12):
#             weight[i, j, k] = (1 / d[i, j, k]) ** 2 / np.sum((1 / d[i, j, :]) ** 2)
# np.save('weight_12.npy', weight)

# weight = np.load('/Users/lihaobo/PycharmProjects/data_no2/weight_12.npy')
# weight = weight.reshape([-1, 12])
# weight = np.float32(weight)
# aqms = np.delete(aqms, [0, 4], 1)
# aqms = np.float32(aqms[:, :])
# s_distribution = np.dot(weight, aqms.transpose())
# torch.save(torch.from_numpy(s_distribution), 'aqms_after_IDW_12_12.pt')

aqms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW_12_12.pt').view(240, 304, -1)
adms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/data_adms.pt').view(-1, 240, 305)[:, :, :304].float()
adms = adms.permute(1, 2, 0)
diff = adms - aqms * 1.88
torch.save(diff, 'diff_12.pt')
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, aqms[:, :, 10], 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, adms[:, :, 10], 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, diff[:, :, 10], 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
plt.show()
pass


