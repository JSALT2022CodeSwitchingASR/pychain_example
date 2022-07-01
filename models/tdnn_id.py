import torch
from torch import nn
import torch.nn.functional as F


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


# TDNN-FNET
class TDNN_FNET(nn.Module):
    def __init__(self, in_dim, out_dim, tdnn_num_layers, tdnn_out_dim, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout, residual=False, fnet_activation=nn.GELU):
        super().__init__()
        self.layers = [TDNN(in_dim, tdnn_out_dim, tdnn_num_layers, hidden_dims[:tdnn_num_layers], kernel_sizes, strides, dilations, dropout, residual)]
        #self.layers += [FNetEncoderBlock(hidden_dims[tdnn_num_layers -1], hidden_dims[i], dropout, fnet_activation()) for i in range(tdnn_num_layers, len(hidden_dims))]
        self.tdnn_fnet = nn.ModuleList(self.layers)
        self.id = nn.Linear(hidden_dims[fnet_num_layers -1], hidden_dims[fnet_num_layers -1])
        self.linear = nn.Linear(hidden_dims[tdnn_num_layers -1], out_dim) #init ?
        # self.layers.append(nn.Tanh()) as added in the paper


    def forward(self, x, x_lengths):
        for layer in (self.tdnn_fnet):
            x, x_lengths = layer(x, x_lengths)
        x = self.id(x)
        x = self.linear(x)
        return x, x_lengths



if __name__ == "__main__":

    # model parameters
    in_dim = 39
    tdnn_out_dim = 100
    tdnn_num_layers = 3
    fnet_num_layers = 3
    out_dim = 10
    hidden_dims = [378, 378, 378, 3072, 3072, 3072]
    dropout = 0.1
    kernel_sizes = [3, 3, 3]
    dilations = [2, 2, 2]
    strides = [2, 2, 2]
  
    net = TDNN_FNET(in_dim, out_dim, tdnn_out_dim, tdnn_num_layers, fnet_num_layers, hidden_dims, kernel_sizes, strides, dilations, dropout) 
    print(net)

    # example
    input = torch.randn(2, 8, in_dim)
    x_lengths = torch.IntTensor([8, 6])
    net(input, x_lengths)
