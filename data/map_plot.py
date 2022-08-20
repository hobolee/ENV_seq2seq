"""
@date        :    20220525
@author      :    Li Haobo
@Description :    cal the index of stations in adms. plot adms data. plot wrf data.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import lib
import numpy as np
import torch
import pandas as pd

# locate stations' location
# station_index = []
# station_loc = [[22.2819, 114.1822], [22.2833, 114.1557], [22.2868, 114.1429], [22.2845, 114.2169], [22.3586, 114.1271], [22.3147, 114.2233], [22.324, 114.166], [22.4968, 114.1284], [22.378, 114.182], [22.3315, 114.1567], [22.2475, 114.16], [22.4524, 114.162], [22.4728, 114.3583], [22.3177, 114.2594], [22.3733, 114.1121], [22.3908, 113.9767], [22.2903, 113.9411], [22.4467, 114.0203]]
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:73200].reshape([240, 305])
# lat = lng_lat['lats'][:73200].reshape([240, 305])
# lon = lon[0, :]
# lat = lat[:, 0]
# for i in range(len(station_loc)):
#     tmp = abs(lat - station_loc[i][0])
#     tmp_lat = list(np.where(tmp == min(tmp)))[0]
#     tmp = abs(lon - station_loc[i][1])
#     tmp_lon = list(np.where(tmp == min(tmp)))[0]
#     station_index.append([tmp_lat[0], tmp_lon[0]])
# print(station_index)

# ADMS 73200
# stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
#                 [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
# data_2019_01 = torch.load('/Users/lihaobo/PycharmProjects/data_no2/adms_after_cor.pt')[:, :, 96+14]
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
# lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data_2019_01, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# for i in range(14):
#     if i == 4:
#         continue
#     plt.scatter(lon[0, stations[i][1]], lat[stations[i][0], 0], c='r')
# plt.scatter(lon[0, 150], lat[150, 0], c='w')
# plt.show()

# # aqms 73200
# data_2019_01 = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')[:, :, 96+14]
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
# lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data_2019_01*1.88, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# for i in range(14):
#     plt.scatter(lon[0, stations[i][1]], lat[stations[i][0], 0], c='r')
# plt.scatter(lon[0, 150], lat[150, 0], c='w')
# plt.show()

# diff 73200
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

# wrf
wrf_path = '/Users/lihaobo/Downloads/wrfout_d04_2019-01-01_12:00:00'
wrf_data_path = '/Users/lihaobo/PycharmProjects/data_no2/wrf_after_cor.pt'
ds = lib.nc_reader(wrf_path)
lon = ds.ds['XLONG'][:].squeeze()[35:95, 100:176]
lat = ds.ds['XLAT'][:].squeeze()[35:95, 100:176]
wrf = torch.load('/Users/lihaobo/PycharmProjects/data_no2/wrf_after_cor.pt')
u10 = wrf[1000, 5, :, :]
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, u10, 60, transform=ccrs.PlateCarree())
ax.coastlines()
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cbar = fig.colorbar(cf, ax=ax, shrink=1)
plt.show()


# ADMS all
# ds = lib.nc_reader('/Volumes/8T/AQMS2/2019/201902/20190201/2019020110.nc')
# data = np.array(ds.read_value('NO2'))[:1200000].reshape([1000, 1200])
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:1200000].reshape([1000, 1200])
# lat = lng_lat['lats'][:1200000].reshape([1000, 1200])
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# plt.show()

# # wrf
# stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
#                 [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
# data_2019_01 = np.load('/Users/lihaobo/PycharmProjects/ENV/NO2/wrf_t2.npy').squeeze()
# lat = np.load('/Users/lihaobo/PycharmProjects/ENV/NO2/wrf_lat.npy').squeeze()
# lon = np.load('/Users/lihaobo/PycharmProjects/ENV/NO2/wrf_lon.npy').squeeze()
# wll = pd.read_csv('/Users/lihaobo/wrf/settings/wrf_idx.csv', index_col=0, na_filter=False).values
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data_2019_01, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# for i in range(len(wll)):
#     plt.scatter(lon[wll[i][0], wll[i][1]], lat[wll[i][0], wll[i][1]], c='r')
# plt.show()

pass
