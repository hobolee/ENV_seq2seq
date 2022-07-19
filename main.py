import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convgru_encoder_params2, convgru_decoder_params2, convgru_encoder_params1, convgru_decoder_params1
from data.adms import ADMS
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse

TIMESTAMP = "2022-07-15T00-00-00_multi"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=2,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-5, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=72,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=1,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
parser.add_argument('-mode', default='local', type=str, help='local or server')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

trainFolder = ADMS(is_train=True,
                   root=r'C:\Users\lihaobo\Downloads\data\data_no2',
                   mode='train')
validFolder = ADMS(is_train=False,
                   root=r'C:\Users\lihaobo\Downloads\data\data_no2',
                   mode='valid')
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

encoder_params1 = convgru_encoder_params1
decoder_params1 = convgru_decoder_params1
encoder_params2 = convgru_encoder_params2
decoder_params2 = convgru_decoder_params2

def train():
    '''
    main function to run the training
    '''
    encoder1 = Encoder(encoder_params1[0], encoder_params1[1])
    decoder1 = Decoder(decoder_params1[0], decoder_params1[1])
    encoder2 = Encoder(encoder_params2[0], encoder_params2[1])
    decoder2 = Decoder(decoder_params2[0], decoder_params2[1])
    net = ED(encoder1, encoder2, decoder1, decoder2)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'), map_location=torch.device('cpu'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    # valid_loss = 10000
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, input_decoder) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device).squeeze()  # B,S,C,H,W
            # input_decoder = input_decoder.to(device)
            # input_decoder = inputs.squeeze(dim=2)
            input_decoder = None
            optimizer.zero_grad()
            net.train()
            pred = net(inputs, input_decoder)[:, -1, :, :, :].squeeze()  # B,S,C,H,W
            # pred = net(inputs, input_decoder).squeeze()  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item()
            train_losses.append(loss_aver)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, input_decoder) in enumerate(t):
                inputs = inputVar.to(device)
                label = targetVar.to(device).squeeze()
                # input_decoder = input_decoder.to(device)
                # input_decoder = inputs.squeeze(dim=2)
                input_decoder = None
                # pred = net(inputs, input_decoder).squeeze()
                pred = net(inputs, input_decoder)[:, -1, :, :, :].squeeze()
                loss = lossfunction(pred, label)
                loss_aver = loss.item()
                # record validation loss
                valid_losses.append(loss_aver)
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss, model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
