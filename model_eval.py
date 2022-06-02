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
# import cartopy.crs as ccrs
from tensorboardX import SummaryWriter


def eval():
    '''
        eval the model
        :return: save the pred_list, label_list, train_loss, valid_loss
        '''
    TIMESTAMP = "2022-06-01T00-00-00"
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
                       root='/Users/lihaobo/PycharmProjects/ENV_prediction/NO2/',
                       n_frames_input=args.frames_input,
                       n_frames_output=args.frames_output)
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
    res = [pred_list, label_list]
    np.save('eval_result2', res)


def eval_plot():
    '''
    plot the ture and predict value.
    plot the loss curve
    :return:
    '''
    result = np.load('eval_result2.npy', allow_pickle=True)
    pred_list = result[0]
    label_list = result[1]
    pred = pred_list[:, :, 100]
    label = label_list[:, :, 100]
    lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
    lon = lng_lat['lngs'][:73200].reshape([240, 305])[:, :304]
    lat = lng_lat['lats'][:73200].reshape([240, 305])[:, :304]
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = plt.contourf(lon, lat, pred, 60, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = fig.colorbar(cf, ax=ax, shrink=1)
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = plt.contourf(lon, lat, label, 60, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = fig.colorbar(cf, ax=ax, shrink=1)
    plt.show()


if __name__ == "__main__":
    eval()
    # eval_plot()
    # eval_adms_station()
