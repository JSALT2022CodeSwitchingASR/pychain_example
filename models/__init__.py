# Copyright (c) Yiwen Shao

# Apache 2.0

from .tdnn import TDNN, TDNN_MFCC
from .rnn import RNN
from .tdnn_lstm import TDNNLSTM
from .fnet import FNET


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              activation, linear_fnet_dim=3072, kernel_sizes=None, strides=None, dilations=None, bidirectional=True, dropout=0, residual=False):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU', 'TDNN-LSTM', 'TDNN-MFCC', 'FNET']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    if arch == 'TDNN':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN(in_dim, out_dim, num_layers,
                     hidden_dims, kernel_sizes, strides, dilations, dropout, residual)

    elif arch == 'TDNN-MFCC':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN')
        model = TDNN_MFCC(in_dim, out_dim, num_layers,
                          hidden_dims, kernel_sizes, strides, dilations, dropout)

    elif arch == 'TDNN-LSTM':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN-LSTM')
        model = TDNNLSTM(in_dim, out_dim, num_layers, hidden_dims, kernel_sizes,
                         strides, dilations, bidirectional, dropout, residual)
    elif arch == 'FNET':
        if not hidden_dims or not dropout or not num_layers:
            raise ValueError(
                'Please specify hidden_dims, dropout and num_layers for FNET')
        model = FNET(num_layers, in_dim, linear_fnet_dim, dropout, out_dim, activation)
    
    else:
        # we simply use same hidden dim for all rnn layers
        hidden_dim = hidden_dims[0]
        model = RNN(in_dim, out_dim, num_layers, hidden_dim,
                    arch, bidirectional, dropout)

    return model
