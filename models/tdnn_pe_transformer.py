import math
import torch
from torch import nn
import torch.nn.functional as F

# testing tdnn + + positional encoding + fnet

# TDNN blocks

class tdnn_bn_relu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1):
        super(tdnn_bn_relu, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = dilation * (kernel_size - 1) // 2
        self.dilation = dilation
        self.tdnn = nn.Conv1d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def output_lengths(self, in_lengths):
        out_lengths = (
            in_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) +
            self.stride - 1
        ) // self.stride
        return out_lengths

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (N, F, T)
        x = self.tdnn(x)
        x = self.bn(x)
        x = self.relu(x)
        x_lengths = self.output_lengths(x_lengths)
        return x, x_lengths



class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden_dims, kernel_sizes, strides, dilations,
                 dropout=0, residual=False):
        super(TDNN, self).__init__()
        assert len(kernel_sizes) == num_layers
        assert len(strides) == num_layers
        assert len(dilations) == num_layers
        self.dropout = dropout
        self.residual = residual
        self.num_layers = num_layers
        self.tdnn = nn.ModuleList([
            tdnn_bn_relu(
                in_dim if layer == 0 else hidden_dims[layer - 1],
                hidden_dims[layer], kernel_sizes[layer],
                strides[layer], dilations[layer],
            )
            for layer in range(num_layers)
        ])
        #self.final_layer = nn.Linear(hidden_dims[-1], out_dim, True) 

    def forward(self, x, x_lengths):
        assert len(x.size()) == 3  # x is of size (B, T, D)
        # turn x to (B, D, T) for tdnn/cnn input
        x = x.transpose(1, 2).contiguous()
        for i in range(len(self.tdnn)):
            # apply Tdnn
            if self.residual and i > 0:  # residual starts from the 2nd layer
                prev_x = x
            x, x_lengths = self.tdnn[i](x, x_lengths)
            x = x + prev_x if (self.residual and i >
                               0 and x.size(2) == prev_x.size(2)) else x
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()  # turn it back to (B, T, D)
        #x = self.final_layer(x)
        return x, x_lengths


# FNET blocks

class ResidualFourierTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.real(torch.fft.fft2(x)) + x


class ResidualFeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, activation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation, #nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.layers(x) + x

class FNetEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, activation):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualFourierTransformLayer(),
            nn.LayerNorm(input_dim), #eps=1e-12 initially
            ResidualFeedForwardLayer(input_dim, hidden_dim, dropout_rate, activation),
            nn.LayerNorm(input_dim) #eps=1e-12
        )
        
    def forward(self, x, x_lengths):
        x = self.layers(x)
        return x, x_lengths


# directly taken from https://pytorch.org/tutorials/advanced/ddp_pipeline.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, num_trans_encoders):
            super().__init__()
            self.transformer = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward) for i in range(num_trans_encoders)])

        def forward(self, x):
            for layer in self.transformer:
                x = layer(x)
            return x


class TDNN_PE_TRANSFORMER(nn.Module):
    def __init__(self, in_dim, out_dim, tdnn_num_layers, tdnn_out_dim, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout, nhead, num_trans_encoders, residual=False):
        super().__init__()
        self.tdnn_layers = nn.ModuleList([TDNN(in_dim, tdnn_out_dim, tdnn_num_layers, hidden_dims[:tdnn_num_layers], kernel_sizes, strides, dilations, dropout, residual)])
        self.position = PositionalEncoding(hidden_dims[tdnn_num_layers - 1], dropout)
        #self.transformer = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dims[tdnn_num_layers - 1], nhead=nhead, dim_feedforward=hidden_dims[tdnn_num_layers]) for i in range(num_trans_encoders)])
        self.transformer = TransformerEncoder(d_model=hidden_dims[tdnn_num_layers - 1], nhead=nhead, dim_feedforward=hidden_dims[tdnn_num_layers], num_trans_encoders=num_trans_encoders)
        self.linear = nn.Linear(hidden_dims[tdnn_num_layers -1], out_dim)
        # self.layers.append(nn.Tanh()) as added in the paper


    def forward(self, x, x_lengths):
        for layer in (self.tdnn_layers):
            x, x_lengths = layer(x, x_lengths)
        x = self.position(x)
        x = self.transformer(x)	
        x = self.linear(x)
        return x, x_lengths



