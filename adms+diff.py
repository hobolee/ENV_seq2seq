import numpy as np
import torch
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
aqms = np.load('aqms_after_interpolation.npy', allow_pickle=True)[0, :]
adms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/data_adms.pt').view(-1, 240, 305)[:, :, :304].float()[0, :, :]
weight = np.load('weight.npy')
weight = weight.reshape([-1, 14])
stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181], [105, 169],
            [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]

diff = []
for i in range(14):
    diff.append(adms[stations[i][0], stations[i][1]] - aqms[i] * 1.88)

diff_map = np.dot(weight, diff).reshape([240, 304, -1]).astype(float).squeeze()
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, diff_map, 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, adms, 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
diff_map = np.dot(weight, diff).reshape([240, 304, -1]).astype(float).squeeze()
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, adms-diff_map, 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
plt.show()
pass