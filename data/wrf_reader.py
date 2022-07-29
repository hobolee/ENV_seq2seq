import sys

import torch
from netCDF4 import Dataset
import numpy as np
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import pandas as pd
import os

data = np.load('/Users/lihaobo/PycharmProjects/data_no2/wrf.npy')
data[:, 2] /= 100
data[:, 3] *= 10
data[:, 4] /= 10000
data = torch.from_numpy(data)
data = np.float32(data)
torch.save(data, 'wrf_after_cor.pt')


# wrf_path = '/Volumes/8T/wrf/2019/201901/2019010112/wrfout_d04_2019-01-01_12:00:00'
# wrf = Dataset(wrf_path)
# lon = wrf.variables['XLONG'][:].squeeze()[35:95, 100:176]
# lat = wrf.variables['XLAT'][:].squeeze()[35:95, 100:176]
# u = wrf.variables['U10'][:].squeeze()[35:95, 100:176]
# v = wrf.variables['V10'][:].squeeze()[35:95, 100:176]
# temp = wrf.variables['T2'][:].squeeze()[35:95, 100:176]
# q_vapor = wrf.variables['QVAPOR'][0, 0, :, :].squeeze()[35:95, 100:176]
# pblh = wrf.variables['PBLH'][:].squeeze()[35:95, 100:176]
# psfc = wrf.variables['PSFC'][:].squeeze()[35:95, 100:176]
# lu = wrf.variables['LU_INDEX'][:].squeeze()[35:95, 100:176]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, pblh, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()



def _get_variables(current):
    ar = np.zeros((1, 6, 60, 76))
    a = Dataset(FILEHEAD + current.strftime('%Y-%m-%d_%H:%M:%S'))
    ar[0, 0, :, :] = a.variables['U10'][:].squeeze()[35:95, 100:176]
    ar[0, 1, :, :] = a.variables['V10'][:].squeeze()[35:95, 100:176]
    ar[0, 2, :, :] = a.variables['T2'][:].squeeze()[35:95, 100:176]
    ar[0, 3, :, :] = a.variables['QVAPOR'][0, 0, :, :].squeeze()[35:95, 100:176]
    ar[0, 4, :, :] = a.variables['PSFC'][:].squeeze()[35:95, 100:176]
    ar[0, 5, :, :] = a.variables['LU_INDEX'][:].squeeze()[35:95, 100:176]
    a.close()
    return ar

WRF_PATH = '/home/dataop/data/nmodel/wrf_fc'
FILEHEAD = 'wrfout_d04_'
ONE_DAY_DELTA = pd.Timedelta(days=1)


start = '20190101'
end = '20220101'
start = pd.to_datetime(start, format='%Y%m%d')
end = pd.to_datetime(end, format='%Y%m%d')
di = pd.date_range(start + pd.Timedelta(hours=12), end + pd.Timedelta(hours=11), freq='H')
dir_time = start - pd.Timedelta(days=1)
data = np.zeros((1, 6, 60, 76))
for current in di:
    # advance dir_time by one day when current is 12:00:00
    if current.hour == 12:
        dir_time += ONE_DAY_DELTA
        print(dir_time)
    file_path = os.path.join(WRF_PATH, str(dir_time.year),
                          dir_time.strftime('%Y%m'),
                          dir_time.strftime('%Y%m%d') + '12')
    os.chdir(file_path)
    li = _get_variables(current)
    data = np.concatenate((data, li), axis=0)
data = data[1:, ...]


np.save('/home/lihaobo/ENV_seq2seq/data/wrf2020.npy', data)
print('done')
