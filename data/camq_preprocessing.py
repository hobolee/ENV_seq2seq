import os.path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch.nn.functional as F
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def cal_IOA(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    label_mean = np.mean(label)
    numerator = sum((pred - label) ** 2)
    denominator = sum((abs(pred - label_mean) + abs(label - label_mean)) ** 2)
    return 1 - numerator / denominator

station_name = ['CB_R',	'CL_R',	'CW_A',	'EN_A',	'KC_A',	'KT_A',	'MKaR',	'ST_A',	'SP_A',	'TP_A',	'MB_A',	'TW_A',	'TC_A',	'YL_A']
lng_lat = np.load('/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]

# read cmaq from csv file
# cmaq = np.zeros([8136, 14])
# root = '/Users/lihaobo/PycharmProjects/data_no2/cmaq'
# for i, name in enumerate(station_name):
#     file_name = 'NO2-' + name + '.csv'
#     df = pd.read_csv(os.path.join(root, file_name), header=2, names=['time', 'obs', 'camx', 'cmaq', 'naqpms'])
#     a = df['cmaq'].values.reshape([-1])
#     cmaq[:, i] = df['cmaq'].values
#
# cmaq = torch.from_numpy(cmaq)
# torch.save(cmaq, 'cmaq.pt')

# valid cmaq
root = '/Users/lihaobo/PycharmProjects/data_no2'
path = os.path.join(root, 'aqms_after_IDW.pt')
aqms = torch.load(path).float()[171, 171, 12:] #, -8136:]
cmaq = torch.load('cmaq.pt')[:, 0]
cmaq = np.array(cmaq)
# x = np.arange(8136)
# fig = plt.figure()
# plt.plot(x, cmaq, 'b')
# plt.scatter(x, aqms, c='r')
# cmaq = torch.from_numpy(cmaq)
# loss = F.mse_loss(aqms, cmaq)
# print(np.sqrt(loss))
# print(cal_IOA(aqms, cmaq))
# print(torch.mean(cmaq))
# print(torch.mean(aqms))

#acf
aqms = np.array(aqms)
plot_acf(aqms, lags=24*30)
plt.show()



# IDW
# weight = np.load('/Users/lihaobo/PycharmProjects/data_no2/weight.npy')
# weight = weight.reshape([-1, 14])
# weight = np.float32(weight)
#
# cmaq = torch.load('cmaq.pt')
# cmaq = np.array(cmaq)
# cmaq = np.float32(cmaq)
# idw = np.dot(weight, cmaq.transpose()).reshape([240, 304, -1])
# torch.save(torch.from_numpy(idw), 'cmaq_after_IDW.pt')
#
# # valid IDW
# cmaq = torch.load('cmaq_after_IDW.pt')
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, cmaq[:, :, -744], 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
pass
