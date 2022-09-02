import os.path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


station_name = ['CB_R',	'CL_R',	'CW_A',	'EN_A',	'KC_A',	'KT_A',	'MKaR',	'ST_A',	'SP_A',	'TP_A',	'MB_A',	'TW_A',	'TC_A',	'YL_A']
lng_lat = np.load('/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]

# read cmaq from csv file
# cmaq = np.zeros([744, 14])
# root = '/Users/lihaobo/PycharmProjects/data_no2/cmaq'
# for i, name in enumerate(station_name):
#     file_name = name + '.csv'
#     df = pd.read_csv(os.path.join(root, file_name), header=2, names=['time', 'obs', 'camx', 'cmaq', 'naqpms'])
#     a = df['cmaq'].values.reshape([-1])
#     cmaq[:, i] = df['cmaq'].values
#
# cmaq = torch.from_numpy(cmaq)
# torch.save(cmaq, 'cmaq.pt')

# valid cmaq
# root = '/Users/lihaobo/PycharmProjects/data_no2'
# path = os.path.join(root, 'aqms_after_IDW.pt')
# aqms = torch.load(path).float()[78, 182, -744:]
# cmaq = torch.load('cmaq.pt')
# fig = plt.figure()
# plt.plot(cmaq[:, 0], 'r')
# plt.plot(aqms, 'b')

# IDW
# weight = np.load('/Users/lihaobo/PycharmProjects/data_no2/weight.npy')
# weight = weight.reshape([-1, 14])
#
# cmaq = torch.load('cmaq.pt')
# cmaq = np.array(cmaq)
# idw = np.dot(weight, cmaq.transpose()).reshape([240, 304, -1]).astype(float)
# torch.save(torch.from_numpy(idw), 'cmaq_after_IDW.pt')

# valid IDW
cmaq = torch.load('cmaq_after_IDW.pt')
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, cmaq[:, :, 10], 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cf, ax=ax, shrink=1)
pass
