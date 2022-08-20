import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np


def plot(a1, a2):
    lng_lat = np.load('/Users/lihaobo/PycharmProjects/data_no2/lnglat-no-receptors.npz')
    lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
    lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]

    fig = plt.figure(figsize=(16, 6))
    ax1 = plt.axes([0.03, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    cf1 = plt.contourf(lon, lat, a1, 50, transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax2 = plt.axes([0.46, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    cf2 = plt.contourf(lon, lat, a2, 50, transform=ccrs.PlateCarree())

    cax = plt.axes([0.92, 0.1, 0.025, 0.8])
    cbar = fig.colorbar(cf1, ax=[ax1, ax2], shrink=1, cax=cax)

weight = np.load('/Users/lihaobo/PycharmProjects/data_no2/weight.npy').reshape([-1, 14])
weight = np.float32(weight)
stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
                [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
aqms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')
adms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/data_adms.pt')
adms = adms.view(-1, 240, 305)[:, :, :304]
adms = adms.permute(1, 2, 0)

adms_mean = torch.mean(adms, 2, keepdim=True)

aqms_station = aqms[stations[0][0], stations[0][1], :].view(-1, 1)
adms_station = adms_mean[stations[0][0], stations[0][1], :].view(-1, 1)
for i in stations:
    if i == [78, 182]:
        continue
    aqms_station = torch.concat((aqms_station, aqms[i[0], i[1], :].view(-1, 1)), 1)
    adms_station = torch.concat((adms_station, adms_mean[i[0], i[1], :].view(-1, 1)), 1)

diff = torch.add(aqms_station * 1.88, adms_station, alpha=-1).permute(1, 0)
distribution = np.dot(weight, diff)
distribution = distribution.reshape([240, 304, -1])
distribution = torch.from_numpy(distribution)

aqms_cor = torch.add(distribution, adms_mean)
plot(aqms_cor[:, :, 0], adms_mean[:, :, 0])
plot(aqms_cor[:, :, 20], adms_mean[:, :, 0])
plot(aqms_cor[:, :, 200], adms_mean[:, :, 0])
plot(aqms_cor[:, :, 2000], adms_mean[:, :, 0])
plot(aqms_cor[:, :, 20000], adms_mean[:, :, 0])
torch.save(aqms_cor, '/Users/lihaobo/PycharmProjects/data_no2/aqms_after_cor.pt')

pass

