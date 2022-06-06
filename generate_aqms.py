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


lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]

# cal weight
# weight = np.zeros([240, 304, 14])
# d = np.zeros([240, 304, 14])
# for i in range(240):
#     print(i)
#     for j in range(304):
#         for k in range(14):
#             d[i, j, k] = np.linalg.norm(np.array((lat[i, j], lon[i, j])) - DI.location[:, k].astype(float))
#         for k in range(14):
#             weight[i, j, k] = (1 / d[i, j, k]) ** 2 / np.sum((1 / d[i, j, :]) ** 2)
# np.save('weight.npy', weight)

weight = np.load('weight.npy')
weight = weight.reshape([-1, 14])
# for i in range(0, 2):
#     ii = i + 8
#     print(ii)
#     if ii == 8:
#         aqms_p = aqms[ii*3000:, :]
#         s_distribution = np.dot(weight, aqms_p.transpose())
#         break
#     else:
#         aqms_p = aqms[ii*3000:(ii+1)*3000, :]
#     print('calculating')
#     tmp = np.dot(weight, aqms_p.transpose())
#     if i == 0:
#         s_distribution = tmp
#     else:
#         s_distribution = np.concatenate((s_distribution, tmp), axis=1)
# s_distribution = s_distribution.reshape([240, 304, -1]).astype(float)
# torch.save(torch.from_numpy(s_distribution), 'aqms_after_IDW5.pt')
# print('saving done')
# aqms1 = torch.load('aqms_after_IDW1.pt').float()
# aqms2 = torch.load('aqms_after_IDW2.pt').float()
# aqms3 = torch.load('aqms_after_IDW3.pt').float()
# aqms4 = torch.load('aqms_after_IDW4.pt').float()
# aqms5 = torch.load('aqms_after_IDW5.pt').float()
# aqms = torch.cat((aqms1, aqms2, aqms3, aqms4, aqms5), 2)
# torch.save(aqms, 'aqms_after_IDW_float.pt')
aqms = torch.load('aqms_after_IDW.pt')
adms = torch.load('/Users/lihaobo/PycharmProjects/ENV/NO2/data_adms.pt').view(-1, 240, 305)[:, :, :304].float()
adms = adms.permute(1, 2, 0)
diff = adms - aqms * 1.88
torch.save(diff, 'diff.pt')
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