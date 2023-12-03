import torch
import numpy as np
from torch import nn, rand
from torch import optim
from omnianomaly.planarNF import PlanarNormalizingFlow

from omnianomaly.RecurrentDistribution import RecurrentDistribution
from omnianomaly.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal, MultivariateNormal
from omnianomaly.lgssm import LinearGaussianStateSpaceModel
from data_config import *

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_dim, dense_dim, freeze_layer: str = None):
        print('Encoder', input_dim, output_dim, rnn_dim, dense_dim)
        super(Encoder, self).__init__()

        self._rnn = nn.Sequential(
          nn.GRU(input_size=input_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True)
        )
        self._linear = nn.Sequential(
            nn.Linear(rnn_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU()
        )

        self._mean = nn.Linear(dense_dim + output_dim, output_dim)
        self._std = nn.Sequential(
            nn.Linear(dense_dim + output_dim, output_dim),
            nn.Softplus()
        )


    def forward(self, x):
        e_output, e_hidden = self._rnn(x)
        e_output = self._linear(e_output)
        return e_output

        
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_dim, dense_dim, freeze_layer: str = None):
        print('Decoder', input_dim, output_dim, rnn_dim, dense_dim)
        super(Decoder, self).__init__()

        self._rnn = nn.Sequential(
          nn.GRU(input_size=input_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True)
        )
        self._linear = nn.Sequential(
            nn.Linear(rnn_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU()
        )

        self._mean = nn.Linear(dense_dim, output_dim)

        self._std = nn.Sequential(
            nn.Linear(dense_dim, output_dim),
            nn.Softplus()
        )
        # self.apply(init_a_layer)
        
    
    def forward(self, x):
        e_output, e_hidden = self._rnn(x)
        # print(f"output:{e_output.shape} hidden:{e_hidden.shape}")
        e_output = self._linear(e_output)
        _mean = self._mean(e_output)
        _std = self._std(e_output)
        return _mean, _std

class VAE_GRU(nn.Module):
    def __init__(self, input_dim, n_sample=10, window_size=60, z_dim=3, freeze_layer: str = None):
        super(VAE_GRU, self).__init__()
        self._encoder = Encoder(input_dim, z_dim, global_rnn_dims, global_dense_dims)
        self._decoder = Decoder(z_dim, input_dim, global_rnn_dims, global_dense_dims)
        self._n_sample = n_sample
        self.is_train = True
        self.window_size = window_size
        self.z_dim = z_dim
        self.input_dim = input_dim
        self._device = global_device
        self.flow_num = 1
        self.flows = nn.Sequential(*[PlanarNormalizingFlow(shape=(self.z_dim,)) for _ in range(self.flow_num)])
        self.p_z = LinearGaussianStateSpaceModel(
            transition_matrix=torch.eye(self.z_dim).to(self._device),
            observation_matrix=torch.eye(self.z_dim).to(self._device),
            transition_noise=MultivariateNormal(loc=torch.zeros(self.z_dim).to(self._device), covariance_matrix=torch.eye(self.z_dim).to(self._device)),
            observation_noise=MultivariateNormal(loc=torch.zeros(self.z_dim).to(self._device), covariance_matrix=torch.eye(self.z_dim).to(self._device)),
            init_state_prior=MultivariateNormal(loc=torch.zeros(self.z_dim).to(self._device), covariance_matrix=torch.eye(self.z_dim).to(self._device)),
            num_timesteps=self.window_size,
        )
        # self.flow = PlanarNormalizingFlow(shape=(self.z_dim,))
        




    def forward(self, x):
        input_q = self._encoder(x)
        q_zx = RecurrentDistribution(input_q=input_q, 
            mean_q_mlp=self._encoder._mean,
            std_q_mlp=self._encoder._std,
            z_dim=self.z_dim,
            window_length=self.window_size
            )
        z_sampled = q_zx.sample(n_samples=(self._n_sample, input_q.shape[0],))
        z_sampled = z_sampled.permute(1, 2, 0, 3) # z^0
        z_sampled_flowed, flow_log_det = self.flows((z_sampled, torch.zeros(1).to(global_device))) # z^K 雅可比矩阵

        z_reshape = z_sampled_flowed.reshape(-1, z_sampled_flowed.shape[-2], z_sampled_flowed.shape[-1])
        
        x_mean, x_std = self._decoder(z_reshape)
        x_mean_reshape = x_mean.view(self._n_sample, x.shape[0], self.window_size, self.input_dim)
        x_std_reshape = x_std.view(self._n_sample, x.shape[0], self.window_size, self.input_dim)
        x_std_reshape = torch.clamp(x_std_reshape, min=min_std)
        p_xz = Normal(x_mean_reshape, x_std_reshape)     
        return q_zx, p_xz, z_sampled, z_sampled_flowed, x_mean_reshape, x_std_reshape, flow_log_det, 
