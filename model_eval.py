"""
@date        :    20220527
@author      :    Li Haobo
@Description :    Eval the model
"""

import torch
import argparse
from tqdm import tqdm
import os
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from lib import DataInterpolate
import pandas as pd
from data.adms import ADMS
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
import cartopy.crs as ccrs
from tensorboardX import SummaryWriter
import matplotlib


def eval():
    '''
        eval the model
        :return: save the pred_list, label_list, train_loss, valid_loss
        '''
    TIMESTAMP = "2022-06-06T00-00-00"
    save_dir = './save_model/' + TIMESTAMP
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size')
    parser.add_argument('-frames_input',
                        default=72,
                        type=int,
                        help='sum of input frames')
    parser.add_argument('-frames_output',
                        default=1,
                        type=int,
                        help='sum of predict frames')
    args = parser.parse_args()

    trainFolder = ADMS(is_train=True,
                       root='/Users/lihaobo/PycharmProjects/data_no2/',
                       mode='all')
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)
    device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    lossfunction = nn.MSELoss()

    # to track the validation loss as the model trains
    test_losses = []
    label_list, pred_list = np.zeros([240, 304]), np.zeros([240, 304])

    tb = SummaryWriter()
    with torch.no_grad():
        net.eval()
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar) in enumerate(t):
            if i == 1000:
                break
            inputs = inputVar  # B,S,C,H,W
            label = targetVar.squeeze()  # B,S,C,H,W
            pred = net(inputs)[:, -1, :, :].squeeze()  # B,S,C,H,W
            if i == 0:
                tb.add_graph(net, inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            test_losses.append(loss_aver)
            label = label.numpy()
            pred = pred.numpy()
            label_list = np.dstack((label_list, label))
            pred_list = np.dstack((pred_list, pred))
            t.set_postfix({
                'testloss': '{:.6f}'.format(loss_aver)
            })
        test_loss = np.average(test_losses)
        print_msg = f'test_loss: {test_loss:.6f} '
        print(print_msg)

    tb.flush()
    tb.close()
    res = [pred_list[:, :, 1:], label_list[:, :, 1:]]
    np.save('eval_result_1000', res)


def eval_plot():
    '''
    plot the ture and predict value.
    plot the loss curve
    :return:
    '''
    result = np.load('eval_result_1000.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
    lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
    lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
    cor_list = []
    for i in range(1000):
        aqms = aqms_data[:, :, i + 48 + 23 + 10000]
        # adms = torch.load('/Users/lihaobo/PycharmProjects/data_no2/data_adms.pt')[i+48+23, :].view(240, 305)[:, :304]
        # adms = adms.numpy()
        # diff = torch.load('/Users/lihaobo/PycharmProjects/data_no2/diff.pt')[:, :, i+48+23]
        # diff = diff.numpy()
        pred = pred_list[:, :, i]
        label = label_list[:, :, i]
        pred = pred + aqms * 1.88
        label = label + aqms * 1.88
        pred_vec = pred.flatten()
        label_vec = label.flatten()
        cor_list.append(np.corrcoef(pred_vec, label_vec)[0][1])
        fig = plt.figure(figsize=(16, 6))
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=150)
        ax1 = plt.axes([0.03, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
        cf1 = plt.contourf(lon, lat, pred, 60, transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.set_title('prediction')
        ax1.set_xlabel('lon')
        ax1.set_ylabel('lat')
        ax2 = plt.axes([0.46, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
        cf2 = plt.contourf(lon, lat, label, 60, transform=ccrs.PlateCarree())
        ax2.set_xlabel('lon')
        ax2.set_title('label')
        ax2.coastlines()
        # plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
        cax = plt.axes([0.92, 0.1, 0.025, 0.8])
        cbar2 = fig.colorbar(cf2, ax=[ax1, ax2], shrink=1, cax=cax)
        # plt.show()
        # plt.savefig('figs/a%s' % i)
        plt.close(fig)
    print(np.mean(cor_list))
        # fig = plt.figure()
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # cf = plt.contourf(lon, lat, adms, 60, transform=ccrs.PlateCarree())
        # ax.coastlines()
        # cbar = fig.colorbar(cf, ax=ax, shrink=1)
        # fig = plt.figure()
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # cf = plt.contourf(lon, lat, diff, 60, transform=ccrs.PlateCarree())
        # ax.coastlines()
        # cbar = fig.colorbar(cf, ax=ax, shrink=1)
        # plt.show()


def eval_ts():
    station = [168, 100]
    result = np.load('eval_result_1000.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    pred_cbr = pred_list[station[0], station[1], :]
    label_cbr = label_list[station[0], station[1], :]
    aqms = aqms_data[station[0], station[1], 10000+48+23:10000+48+23+1000]
    pred_cbr = pred_cbr + aqms*1.88
    label_cbr = label_cbr + aqms*1.88
    print(np.corrcoef(pred_cbr, label_cbr))
    r2 = 1 - np.sum((pred_cbr - label_cbr) ** 2) / np.sum((label_cbr - np.mean(label_cbr))**2)
    print(r2)

    plt.figure()
    x = np.arange(1000)
    plt.plot(x, pred_cbr, 'b', x, label_cbr, 'r')
    plt.show()


if __name__ == "__main__":
    # eval()
    eval_plot()
    # eval_ts()
    # eval_adms_station()
