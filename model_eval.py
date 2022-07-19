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
from net_params import convgru_encoder_params2, convgru_decoder_params2, convgru_encoder_params1, convgru_decoder_params1
import cartopy.crs as ccrs
from tensorboardX import SummaryWriter
import matplotlib


def aqms_correction(pred, weight, i):
    stations = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [130, 181],
                [105, 169], [171, 171], [182, 270], [128, 146], [83, 60], [168, 100]]
    aqms_station = np.load('aqms_after_interpolation.npy', allow_pickle=True)[i + 72 + 23, :]
    diff = []
    for j in range(14):
        diff.append(pred[stations[j][0], stations[j][1]] - aqms_station[j] * 1.88)
    diff_map = np.dot(weight, diff).reshape([240, 304, -1]).astype(float).squeeze()[:, :]
    pred = pred - diff_map
    return pred


def cal_cor(pred, label):
    pred_vec = pred.flatten()
    label_vec = label.flatten()
    return np.corrcoef(pred_vec, label_vec)[0][1]


def cal_IOA(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    label_mean = np.mean(label)
    numerator = sum((pred - label) ** 2)
    denominator = sum((abs(pred - label_mean) + abs(label - label_mean)) ** 2)
    return 1 - numerator / denominator


def plot(pred, label, lon, lat, i, mode):
    fig = plt.figure(figsize=(16, 6))
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    ax1 = plt.axes([0.03, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    pred[pred > 150] = 150
    label[label > 150] = 150
    # cf1 = plt.contourf(lon, lat, pred, transform=ccrs.PlateCarree(), levels=range(151))
    cf1 = plt.contourf(lon, lat, pred, 60, transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_title('prediction')
    ax1.set_xlabel('lon')
    ax1.set_ylabel('lat')
    ax2 = plt.axes([0.46, 0.1, 0.455, 0.8], projection=ccrs.PlateCarree())
    # cf2 = plt.contourf(lon, lat, label, transform=ccrs.PlateCarree(), levels=range(151))
    cf2 = plt.contourf(lon, lat, label, 60, transform=ccrs.PlateCarree())
    ax2.set_xlabel('lon')
    ax2.set_title('label')
    ax2.coastlines()
    # plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.92, 0.1, 0.025, 0.8])
    # cbar = fig.colorbar(cf2, ax=[ax1, ax2], shrink=1, cax=cax, ticks=[0, 30, 60, 90, 120, 150])
    # cbar.set_ticklabels(['0', '30', '60', '90', '120', '>150'])
    cbar = fig.colorbar(cf1, ax=[ax1, ax2], shrink=1, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('No2(ppb)')
    if mode == 'show':
        plt.show()
    elif mode == 'save':
        plt.savefig('figs/figs_diff_72to24_norm_relu/a%s' % i)
    plt.close(fig)


def negetive_correction(pred):
    # if pred.min() < 0:
    #     pred -= pred.min()
    pred[pred < 0] = 0.1
    return pred


def mean_corection(pred, label):
    mean_pred = np.mean(pred)
    mean_label = np.mean(label)
    pred = pred - (mean_pred - mean_label)
    return pred


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
    random_seed = 1996
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    TIMESTAMP = "2022-07-15T00-00-00_multi"
    save_dir = './save_model/' + TIMESTAMP
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=2,
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
                       root=r'C:\Users\lihaobo\Downloads\data\data_no2',
                       mode='all')
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    encoder_params1 = convgru_encoder_params1
    decoder_params1 = convgru_decoder_params1
    encoder_params2 = convgru_encoder_params2
    decoder_params2 = convgru_decoder_params2
    encoder1 = Encoder(encoder_params1[0], encoder_params1[1])
    decoder1 = Decoder(decoder_params1[0], decoder_params1[1])
    encoder2 = Encoder(encoder_params2[0], encoder_params2[1])
    decoder2 = Decoder(decoder_params2[0], decoder_params2[1])
    net = ED(encoder1, encoder2, decoder1, decoder2)
    device = torch.device("cuda:0")

    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))#, map_location=torch.device('cpu'))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    lossfunction = nn.MSELoss()
    net.to(device)

    # to track the validation loss as the model trains
    test_losses = []
    label_list, pred_list = np.zeros([1, 240, 304]), np.zeros([1, 240, 304])

    tb = SummaryWriter()
    with torch.no_grad():
        net.eval()
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, input_decoder) in enumerate(t):
            # if i == 1000:
            #     break
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device).squeeze()  # B,S,C,H,W
            # input_decoder = input_decoder.to(device)
            # input_decoder = inputs.squeeze(dim=2)
            input_decoder = None
            # pred = net(inputs, input_decoder).squeeze()  # B,S,C,H,W
            pred = net(inputs, input_decoder)[:, -1, :, :, :].squeeze()
            if i == 0:
                print(pred)
            #     tb.add_graph(net, inputs)
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            test_losses.append(loss_aver)
            label1 = label.to("cpu").numpy().reshape([-1, 1, 240, 304])
            pred1 = pred.to("cpu").numpy().reshape([-1, 1, 240, 304])
            if i == 0:
                label_list = label1
                pred_list = pred1
            else:
                label_list = np.concatenate((label_list, label1), axis=0)
                pred_list = np.concatenate((pred_list, pred1), axis=0)
            t.set_postfix({
                'testloss': '{:.6f}'.format(loss_aver)
            })
        test_loss = np.average(test_losses)
        print_msg = f'test_loss: {test_loss:.6f} '
        print(print_msg)

    tb.flush()
    tb.close()
    res = [pred_list, label_list]
    np.save('eval_result_diff_72to24', res)


def eval_plot():
    '''
    plot the ture and predict value.
    plot the loss curve
    :return:
    '''
    result = np.load('eval_result_diff_72to24.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load(r'C:\Users\lihaobo\Downloads\data\data_no2\aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    lng_lat = np.load(r'C:\Users\lihaobo\Downloads\data\data_no2\lnglat-no-receptors.npz')
    lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
    lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
    weight = np.load('weight.npy')
    weight = weight.reshape([-1, 14])
    ioa_list = []
    mse_before, mse_after, mse_before_n, mse_after_n, mse_before_m, mse_after_m = [], [], [], [], [], []
    for i in range(1000):
        aqms = aqms_data[:, :, i + 72 + 23]
        pred = pred_list[i, 0, :, :]
        label = label_list[i, 0, :, :]

        # pred, label = diff2adms(pred, label, aqms)

        # mse_b_m = cal_mse(pred, aqms)
        # pred = mean_correction(pred, aqms)
        # mse_a_m = cal_mse(pred, label)
        # mse_before_m.append(mse_b_m)
        # mse_after_m.append(mse_a_m)

        # mse_b = cal_mse(pred, label)
        # pred = aqms_correction(pred, weight, i)
        # mse_a = cal_mse(pred, label)
        # mse_before.append(mse_b)
        # mse_after.append(mse_a)
        # mse_b_n = cal_mse(pred, label)
        # pred = negetive_correction(pred)
        # mse_a_n = cal_mse(pred, label)
        # mse_before_n.append(mse_b_n)
        # mse_after_n.append(mse_a_n)
        plt.figure()
        plt.contourf(lon, lat, pred, 50)
        plt.colorbar()
        plt.figure()
        plt.contourf(lon, lat, label, 50)
        plt.colorbar()
        plt.show()
        ioa = cal_IOA(pred, label)
        ioa_list.append(ioa)

        plot(pred, label, lon, lat, i, 'save')
    print(np.mean(ioa_list))
    # print('mse_before_m', np.mean(mse_before_m))
    # print('mse_after_m', np.mean(mse_after_m))
    # print('mse_before', np.mean(mse_before))
    # print('mse_after', np.mean(mse_after))
    print('mse_before_n', np.mean(mse_before_n))
    print('mse_after_n', np.mean(mse_after_n))


def eval_ts():
    result = np.load('eval_result_diff_72to24.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    aqms_data = torch.load(r'C:\Users\lihaobo\Downloads\data\data_no2\aqms_after_IDW.pt')
    aqms_data = aqms_data.numpy()
    weight = np.load('weight.npy')
    weight = weight.reshape([-1, 14])
    cor_list, pred_station, label_station = [], [], []
    # station = [150, 150]
    # station = [78, 182]
    # station = [79, 168]
    station = [120, 154]
    for i in range(1000):
        aqms = aqms_data[:, :, i + 72 + 23]
        pred = pred_list[i, 0, :, :]
        label = label_list[i, 0, :, :]
        pred, label = diff2adms(pred, label, aqms)
        # pred = mean_corection(pred, aqms)
        # pred = negetive_correction(pred)
        # pred = aqms_correction(pred, weight, i)
        pred_station.append(pred[station[0], station[1]])
        label_station.append(label[station[0], station[1]])
    lag = 0
    if lag:
        print(np.corrcoef(pred_station[lag:], label_station[:-lag]))
        print(cal_IOA(pred_station[lag:], label_station[:-lag]))
    else:
        print(np.corrcoef(pred_station, label_station))
        print(cal_IOA(pred_station, label_station))
    plt.figure()
    x = np.arange(1000 - lag)
    if lag:
        plt.plot(x, pred_station[lag:], 'b', x, label_station[:-lag], 'r')
    else:
        plt.plot(x, pred_station, 'b', x, label_station, 'r')
    plt.show()


if __name__ == "__main__":
    # eval()
    eval_plot()
    # eval_ts()
    # eval_adms_station()
