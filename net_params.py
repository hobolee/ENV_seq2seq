from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [16, 16, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [24, 24, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=4, filter_size=5, num_features=16),
        CLSTM_cell(shape=(32,32), input_channels=16, filter_size=5, num_features=24),
        CLSTM_cell(shape=(16,16), input_channels=24, filter_size=5, num_features=24)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [24, 24, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [24, 24, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [24, 4, 3, 1, 1],
            'conv4_leaky_1': [4, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=24, filter_size=5, num_features=24),
        CLSTM_cell(shape=(32,32), input_channels=24, filter_size=5, num_features=24),
        CLSTM_cell(shape=(64,64), input_channels=24, filter_size=5, num_features=16),
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [8, 8, 3, 4, 1]}),
        # OrderedDict({'conv3_leaky_1': [12, 12, 3, 4, 1]}),
    ],

    [
        CGRU_cell(shape=(240, 304), input_channels=4, filter_size=5, num_features=8),
        CGRU_cell(shape=(60, 76), input_channels=8, filter_size=5, num_features=12),
        # CGRU_cell(shape=(15, 19), input_channels=12, filter_size=5, num_features=12)
    ]
]

convgru_decoder_params = [
    [
        # OrderedDict({'deconv1_leaky_1': [12, 12, 6, 4, 1]}),
        OrderedDict({'deconv1_leaky_1': [12, 12, 6, 4, 1]}),
        OrderedDict({
            'conv2_leaky_1': [8, 4, 3, 1, 1],
            'conv3_leaky_1': [4, 1, 1, 1, 0]
        }),
    ],

    [
        # CGRU_cell(shape=(15,19), input_channels=12, filter_size=5, num_features=12),
        CGRU_cell(shape=(60,76), input_channels=12, filter_size=5, num_features=12),
        CGRU_cell(shape=(240,304), input_channels=12, filter_size=5, num_features=8),
    ]
]