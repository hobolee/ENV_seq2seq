from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers


class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ED(nn.Module):

    def __init__(self, encoder1, encoder2, decoder1, decoder2):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2

    def forward(self, input, input_decoder):
        input_low = input[:, :, :, ::2, ::2]
        state1 = self.encoder1(input_low)
        state2 = self.encoder2(input)
        output_low = self.decoder1(state1, input_decoder)
        output_low = output_low[:, -1, :, :, :].squeeze()
        state = list(state2)
        state[1] = torch.concat((state[1], output_low), 1)
        output = self.decoder2(state, input_decoder)
        return output
