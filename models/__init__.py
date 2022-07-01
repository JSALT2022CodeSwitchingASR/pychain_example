# Copyright (c) Yiwen Shao

# Apache 2.0

from .tdnn import TDNN, TDNN_MFCC
from .rnn import RNN
from .tdnn_lstm import TDNNLSTM
from .fnet import FNET
from .tdnn_fnet import TDNN_FNET
from .test_arch import TEST_ARCH
from .tdnn_pe_fnet import TDNN_PE_FNET
from .tdnn_pe_transformer import TDNN_PE_TRANSFORMER


def get_model(in_dim, out_dim, num_layers, hidden_dims, arch,
              fnet_activation, fnet_num_layers=3, tdnn_out_dim=384, tdnn_num_layers=3, kernel_sizes=None, strides=None, dilations=None, bidirectional=True, nhead=4, num_trans_encoders=2, dropout=0, residual=False):
    valid_archs = ['TDNN', 'RNN', 'LSTM', 'GRU', 'TDNN-LSTM', 'TDNN-MFCC', 'FNET', 'TDNN-FNET', 'TDNN_PE_FNET', 'TDNN_PE_TRANSFORMER']
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
        model = FNET(in_dim, out_dim, hidden_dims, dropout, fnet_activation)
    elif arch == 'TDNN-FNET':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN-FNET')
        model = TDNN_FNET(in_dim, out_dim, tdnn_num_layers, tdnn_out_dim, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout, residual, fnet_activation) 
    elif arch == 'TDNN_PE_FNET':
        if not kernel_sizes or not dilations or not strides:
            raise ValueError(
                'Please specify kernel sizes, strides and dilations for TDNN-FNET')
        model = TDNN_PE_FNET(in_dim, out_dim, tdnn_num_layers, tdnn_out_dim, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout, residual, fnet_activation) 
    elif arch == 'TDNN_PE_TRANSFORMER':
         if not nhead or not num_trans_encoders:
            raise ValueError(
                'Please specify number of encoders and attention heads for the Transformer')
         model = TDNN_PE_TRANSFORMER(in_dim, out_dim, tdnn_num_layers, tdnn_out_dim, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout, nhead, num_trans_encoders)

	
    else:
        # we simply use same hidden dim for all rnn layers
        hidden_dim = hidden_dims[0]
        model = RNN(in_dim, out_dim, num_layers, hidden_dim,
                    arch, bidirectional, dropout)

    return model
