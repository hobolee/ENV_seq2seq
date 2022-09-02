import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import torch

# not correct
stations = [[44, 89], [44, 89], [44, 89], [44, 89], [44, 89], [44, 89], [44, 89], [44, 89], [44, 89], [45, 89], [44, 89], [44, 89], [44, 86], [45, 89]]
lat_low =5
lat_high = 65
lon_low = 75
lon_high = 151

cmaq = np.load('/Users/lihaobo/PycharmProjects/data_no2/cmaq202112.npy')[:, :, lat_low:lat_high, lon_low:lon_high]
cmaq_path = '/Users/lihaobo/PycharmProjects/data_no2/GRIDCRO2D.1km'
cmaq_tmp = Dataset(cmaq_path)
lon = cmaq_tmp.variables['LON'][:].squeeze()[lat_low:lat_high, lon_low:lon_high]
lat = cmaq_tmp.variables['LAT'][:].squeeze()[lat_low:lat_high, lon_low:lon_high]

# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, cmaq[0, 0, :, :], 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()

station = [22.2819, 114.1822]
d_min = 10000
for i in range(60):
    for j in range(76):
        d = (lat[i, j] - station[0]) ** 2 + (lon[i, j] - station[1]) ** 2
        if d < d_min:
            d_min = d
            lat_i = i
            lon_i = j

# to see if the two points are close
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, cmaq[0, 0, :, :], 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.scatter(station[1], station[0], c='r')
# plt.scatter(lon[lat_i, lon_i], lat[lat_i, lon_i],  c='b')
# print(lon[lat_i, lon_i], lat[lat_i, lon_i])
# plt.show()

root = '/Users/lihaobo/PycharmProjects/data_no2'
path = os.path.join(root, 'aqms_after_IDW.pt')
aqms = torch.load(path).float()[78, 182, -720:]

fig = plt.figure()
plt.plot(cmaq[:, 0, lat_i, lon_i] * 25 * 50)
plt.plot(aqms)


pass