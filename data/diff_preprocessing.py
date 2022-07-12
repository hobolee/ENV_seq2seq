import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch


stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
            [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
weight = np.load('../weight.npy')
weight = np.float32(weight)
weight = weight.reshape([-1, 14])

# data_2019_01 = torch.load('/Users/lihaobo/PycharmProjects/data_no2/diff.pt')[:, :, 96+14]
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
# lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data_2019_01, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# for i in range(14):
#     plt.scatter(lon[0, stations[i][1]], lat[stations[i][0], 0], c='r')
# plt.scatter(lon[0, 150], lat[150, 0], c='w')
# plt.show()

# diff_station = [data_2019_01[i, j].item() for (i, j) in stations]
# distribution = np.dot(weight, diff_station)
# distribution = distribution.reshape([240, 304])
# distribution = data_2019_01 - distribution
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, distribution, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)


diff = torch.load('/Users/lihaobo/PycharmProjects/data_no2/diff.pt')
diff_station = diff[78, 182, :].view([-1, 1])
for i in range(1, 14):
    x, y = stations[i]
    tmp = diff[x, y, :].view([-1, 1])
    diff_station = torch.cat((diff_station, tmp), 1)
diff_station = diff_station.transpose(1, 0)

distribution = np.dot(weight, diff_station)
distribution = distribution.reshape([240, 304, -1])
torch.save(torch.from_numpy(distribution), 'diff_after_cor.pt')


