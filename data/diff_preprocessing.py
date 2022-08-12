import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch


stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
            [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
weight = np.load('/Users/lihaobo/PycharmProjects/data_no2/weight.npy')
weight = np.float32(weight)
# weight = weight.reshape([-1, 12])

# data_2019_01 = torch.load(r'C:\Users\lihaobo\Downloads\data\data_no2\diff.pt')[:, :, 1000]
# lng_lat = np.load(r'C:\Users\lihaobo\Downloads\data\data_no2\lnglat-no-receptors.npz')
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


# diff = torch.load(r'/Users/lihaobo/PycharmProjects/data_no2/diff_12.pt')
# stations = np.delete(stations, [0, 4], 0)
# # weight = np.delete(weight, [0, 4], 1)
# diff_station = diff[79, 168, :].view([-1, 1])
# for i in range(1, 12):
#     x, y = stations[i]
#     tmp = diff[x, y, :].view([-1, 1])
#     diff_station = torch.cat((diff_station, tmp), 1)
# diff_station = diff_station.transpose(1, 0)
#
# distribution = np.dot(weight, diff_station)
# distribution = distribution.reshape([240, 304, -1])
# distribution = diff - distribution
# torch.save(distribution, 'diff_after_cor_12.pt')

# # distribution = torch.load(r'./diff_after_cor_12.pt')
# lng_lat = np.load(r'/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
# plt.figure()
# plt.plot(distribution[79, 168, :], 'b')
# plt.plot(diff[79, 168, :], 'r')
# lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
# lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, distribution[:, :, 1000], 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# # for i in range(14):
# #     plt.scatter(lon[0, stations[i][1]], lat[stations[i][0], 0], c='r')
# # plt.scatter(lon[0, 150], lat[150, 0], c='w')
# plt.show()

adms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/data_adms.pt').view(-1, 240, 305)[:, :, :304]
adms = adms.permute(1, 2, 0)
aqms = np.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_interpolation.npy', allow_pickle=True).astype(float)
aqms = np.float32(aqms)
aqms = torch.from_numpy(aqms)
aqms = aqms.permute(1, 0)
adms_station = adms[79, 168, :].view([-1, 1])
for i in range(1, 14):
    x, y = stations[i]
    tmp = adms[x, y, :].view([-1, 1])
    adms_station = torch.cat((adms_station, tmp), 1)
adms_station = adms_station.transpose(1, 0)
diff = adms_station - aqms * 1.88

weight = np.reshape(weight, (-1, 14))
distribution = np.dot(weight, diff)
distribution = distribution.reshape([240, 304, -1])
distribution = adms - distribution
# torch.save(distribution, 'adms_after_cor.pt')

# distribution = torch.load(r'./diff_after_cor_12.pt')
lng_lat = np.load(r'/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
plt.figure()
plt.plot(distribution[120, 152, :], 'b')
# plt.plot(adms[79, 168, :], 'r')
plt.plot(aqms[4, :] * 1.88, 'r')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, distribution[:, :, 1000], 60, transform=ccrs.PlateCarree())
ax.coastlines()
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cbar = fig.colorbar(cf, ax=ax, shrink=1)
# for i in range(14):
#     plt.scatter(lon[0, stations[i][1]], lat[stations[i][0], 0], c='r')
# plt.scatter(lon[0, 150], lat[150, 0], c='w')
plt.show()
