import sys
# import torch
from netCDF4 import Dataset
import numpy as np
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import pandas as pd
import os

cmaq_path = '/Users/lihaobo/Downloads/1km/CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.2021002'
cmaq = Dataset(cmaq_path)
lon = cmaq.variables['XLONG'][:].squeeze()[35:95, 100:176]
# lat = cmaq.variables['XLAT'][:].squeeze()[35:95, 100:176]
# u = cmaq.variables['U10'][:].squeeze()[35:95, 100:176]
# v = cmaq.variables['V10'][:].squeeze()[35:95, 100:176]
# temp = cmaq.variables['T2'][:].squeeze()[35:95, 100:176]
# q_vapor = cmaq.variables['QVAPOR'][0, 0, :, :].squeeze()[35:95, 100:176]
# pblh = cmaq.variables['PBLH'][:].squeeze()[35:95, 100:176]
# psfc = cmaq.variables['PSFC'][:].squeeze()[35:95, 100:176]
# lu = cmaq.variables['LU_INDEX'][:].squeeze()[35:95, 100:176]
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, pblh, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()

def _get_variables(index):
    ar = np.zeros((24, 2, 125, 179))
    a = Dataset(FILEHEAD + index)
    ar[:, 0, :, :] = a.variables['NO2'][:].squeeze()[:, 0, :, :]
    ar[:, 1, :, :] = a.variables['NO2'][:].squeeze()[:, 1, :, :]
    a.close()
    return ar

CMAQ_PATH = '/home/dataop/data/nmodel/cmaq_fc'
FILEHEAD = 'CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.2021'
ONE_DAY_DELTA = pd.Timedelta(days=1)


start = '20211201'
end = '20211231'
start = pd.to_datetime(start, format='%Y%m%d')
end = pd.to_datetime(end, format='%Y%m%d')
di = pd.date_range(start + pd.Timedelta(hours=12), end + pd.Timedelta(hours=11), freq='D')
dir_time = start - pd.Timedelta(days=1)
data = np.zeros((1, 2, 125, 179))
for current in di:
    # advance dir_time by one day when current is 12:00:00
    index = current.dayofyear + 1
    if current.hour == 12:
        dir_time += ONE_DAY_DELTA
        print(dir_time)
    file_path = os.path.join(CMAQ_PATH, str(dir_time.year),
                          dir_time.strftime('%Y%m'),
                          dir_time.strftime('%Y%m%d') + '12', '1km')
    os.chdir(file_path)
    li = _get_variables(str(index))
    data = np.concatenate((data, li), axis=0)
data = data[1:, ...]


np.save('/home/lihaobo/ENV_seq2seq/data/cmaq202112.npy', data)
print('done')
