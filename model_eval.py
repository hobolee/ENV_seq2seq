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


def aqms_correction(pred, weight, i):
    stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
                [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
    aqms_station = np.load('aqms_after_interpolation.npy', allow_pickle=True)[i, :]
    diff = []
    for j in range(14):
        diff.append(pred[stations[j][0] // 2, stations[j][1] // 2] - aqms_station[j] * 1.88)
    diff_map = np.dot(weight, diff).reshape([240, 304, -1]).astype(float).squeeze()[::2, ::2]
    pred = pred - diff_map
    return pred


def negetive_correction(pred):
    if pred.min() < 0:
        pred -= pred.min()
    return pred


def mean_corection(pred, label):
    mean_pred = np.mean(pred)
    mean_label = np.mean(label)
    pred = pred - (mean_pred - mean_label)
    return pred


def cal_cor(pred, label):
    pred_vec = pred.flatten()
    label_vec = label.flatten()
    return np.corrcoef(pred_vec, label_vec)[0][1]


def plot(pred, label, lon, lat, i, mode):
    fig = plt.figure(figsize=(16, 6))
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    ax1 = plt.axes([0.03, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    pred[pred > 150] = 150
    label[label > 150] = 150
    cf1 = plt.contourf(lon, lat, pred, transform=ccrs.PlateCarree(), levels=range(151))
    ax1.coastlines()
    ax1.set_title('prediction')
    ax1.set_xlabel('lon')
    ax1.set_ylabel('lat')
    ax2 = plt.axes([0.46, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    cf2 = plt.contourf(lon, lat, label, transform=ccrs.PlateCarree(), levels=range(151))
    ax2.set_xlabel('lon')
    ax2.set_title('label')
    ax2.coastlines()
    # plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.92, 0.1, 0.025, 0.8])
    cbar = fig.colorbar(cf2, ax=[ax1, ax2], shrink=1, cax=cax, ticks=[0, 30, 60, 90, 120, 150])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('No2(ppb)')
    # cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(['0', '30', '60', '90', '120', '>150'])
    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig('figs_low/a%s' % i)
    plt.close(fig)


def diff2adms(pred, label, aqms):
    pred = pred + aqms * 1.88
    label = label + aqms * 1.88
    return pred, label


def cal_r2(pred, label):
    r2 = 1 - np.sum((pred - label) ** 2) / np.sum((label - np.mean(label))**2)
    return r2


def cal_mse(pred, label):
    return  (np.square(pred - label)).mean(axis=None)


def eval():
    '''
        eval the model
        :return: save the pred_list, label_list, train_loss, valid_loss
        '''
    TIMESTAMP = "2022-06-09T00-00-00_low"
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
    label_list, pred_list = np.zeros([120, 152]), np.zeros([120, 152])

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
            # if i == 0:
            #     tb.add_graph(net, inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
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
    np.save('eval_result_1000_low', res)


def eval_plot():
    '''
    plot the ture and predict value.
    plot the loss curve
    :return:
    '''
    result = np.load('eval_result_1000_low.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
    lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304][::2, ::2]
    lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304][::2, ::2]
    weight = np.load('weight.npy')
    weight = weight.reshape([-1, 14])
    cor_list = []
    mse_before, mse_after, mse_before_n, mse_after_n = [], [], [], []
    for i in range(1000):
        aqms = aqms_data[:, :, i + 24 + 23]
        pred = pred_list[:, :, i]
        label = label_list[:, :, i]
        mse_b = cal_mse(pred, label)
        pred = aqms_correction(pred, weight, i)
        mse_a = cal_mse(pred, label)
        mse_before.append(mse_b)
        mse_after.append(mse_a)
        mse_b_n = cal_mse(pred, label)
        pred = negetive_correction(pred)
        mse_a_n = cal_mse(pred, label)
        mse_before_n.append(mse_b_n)
        mse_after_n.append(mse_a_n)
        cor = cal_cor(pred, label)
        cor_list.append(cor)
        plot(pred, label, lon, lat, i, 'save')
    print(np.mean(cor_list))
    print('before', np.mean(mse_before))
    print('after', np.mean(mse_after))
    print('before_n', np.mean(mse_before_n))
    print('after_n', np.mean(mse_after_n))


def eval_ts():
    result = np.load('eval_result_1000_low.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load('/Users/lihaobo/PycharmProjects/data_no2/aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    weight = np.load('weight.npy')
    weight = weight.reshape([-1, 14])
    cor_list, pred_station, label_station = [], [], []
    station = [81, 162]
    for i in range(1000):
        pred = pred_list[:, :, i]
        label = label_list[:, :, i]
        # aqms = aqms_data[:, :, i + 24 + 23]
        # pred, label = diff2adms(pred, label, aqms)
        # pred = mean_corection(pred, label)
        pred = aqms_correction(pred, weight, i)
        pred = negetive_correction(pred)
        pred_station.append(pred[station[0]//2, station[1]//2])
        label_station.append(label[station[0]//2, station[1]//2])
    print(np.corrcoef(pred_station[72:], label_station[:-72]))

    plt.figure()
    x = np.arange(928)
    plt.plot(x, pred_station[72:], 'b', x, label_station[:-72], 'r')
    plt.show()


if __name__ == "__main__":
    # eval()
    # eval_plot()
    eval_ts()
    # eval_adms_station()