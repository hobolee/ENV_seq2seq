from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]


convgru_encoder_params0 = [
    [
        OrderedDict({'conv1_leaky_1': [6, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(60, 76), input_channels=16, filter_size=5, num_features=32),
        CGRU_cell(shape=(30, 38), input_channels=32, filter_size=5, num_features=64),
        CGRU_cell(shape=(15, 19), input_channels=64, filter_size=5, num_features=64)
    ]
]

convgru_encoder_params1 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
        OrderedDict({'conv4_leaky_1': [128, 128, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(120, 152), input_channels=16, filter_size=5, num_features=64),
        CGRU_cell(shape=(60, 76), input_channels=64, filter_size=5, num_features=96),
        CGRU_cell(shape=(30, 38), input_channels=96, filter_size=5, num_features=128),
        CGRU_cell(shape=(15, 19), input_channels=128, filter_size=5, num_features=128)
    ]
]

convgru_decoder_params1 = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({'deconv3_leaky_1': [96, 96, 4, 2, 1]}),
        # OrderedDict({'conv4_relu_1': [8, 8, 3, 1, 1]}),
        OrderedDict({
            'conv4_leaky_1': [64, 32, 3, 1, 1],
            'conv5_no_1': [32, 16, 1, 1, 0],
            'conv6_no_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(15, 19), input_channels=192, filter_size=5, num_features=192),
        CGRU_cell(shape=(30, 38), input_channels=192, filter_size=5, num_features=128),
        CGRU_cell(shape=(60, 76), input_channels=128, filter_size=5, num_features=96),
        CGRU_cell(shape=(120, 152), input_channels=96, filter_size=5, num_features=64),
    ]
]
convgru_encoder_params2 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [8, 8, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(240, 304), input_channels=4, filter_size=5, num_features=8),
        CGRU_cell(shape=(120, 152), input_channels=8, filter_size=5, num_features=16),
    ]
]

convgru_decoder_params2 = [
    [
        OrderedDict({'deconv1_leaky_1': [24, 24, 4, 2, 1]}),
        OrderedDict({
            'conv2_leaky_1': [8, 32, 3, 1, 1],
            'conv3_no_1': [32, 128, 1, 1, 0],
            'conv4_no_1': [128, 64, 1, 1, 0],
            'conv5_no_1': [64, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(120, 152), input_channels=24, filter_size=5, num_features=24),
        CGRU_cell(shape=(240, 304), input_channels=24, filter_size=5, num_features=8),
    ]
]
