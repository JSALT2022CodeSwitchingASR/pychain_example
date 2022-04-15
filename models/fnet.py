import torch
from torch import nn


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
        
    def forward(self, x):
        return self.layers(x)


class FNET(nn.Module):
    def __init__(self, n_encoders, input_dim, hidden_dim, dropout_rate, output_dim, activation=nn.GELU):
        super().__init__()
        self.layers = nn.ModuleList([FNetEncoderBlock(input_dim, hidden_dim, dropout_rate, activation()) for i in range(n_encoders)])
        self.layers.append(nn.Linear(input_dim, output_dim)) #init ?
        # self.layers.append(nn.Tanh()) as added in the paper
        
        
    def forward(self, x, x_lengths):
        for layer in (self.layers):
            x = layer(x)
        return x, x_lengths


if __name__ == "__main__":
	# model parameters
	N = 6
	input_dim = 39
	hidden_dim = 3072
	dropout_rate = 0.1
	output_dim = 48

	# example
	batch_size = 16
	seq_length = 1000
	x = torch.rand(batch_size, seq_length, input_dim)

	fnet_encoder = FNET(N, input_dim, hidden_dim, dropout_rate, output_dim, activation)


