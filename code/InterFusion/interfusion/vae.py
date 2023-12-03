import torch
import numpy as np
from torch import nn
from torch import optim
from interfusion.planarNF import PlanarNormalizingFlow

from interfusion.RecurrentDistribution import RecurrentDistribution
from interfusion.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal, MultivariateNormal
# from interfusion.lgssm import LinearGaussianStateSpaceModel
from data_config import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False

act_func = nn.LeakyReLU(negative_slope=0.2)
# act_func = nn.ReLU()

class h_for_qz(nn.Module):
    def __init__(self, input_channel):
        super(h_for_qz, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=input_channel*2, kernel_size=7, stride=2, padding=3),
            act_func,
            # nn.Conv1d(in_channels=input_channel*2, out_channels=input_channel*2, kernel_size=7, stride=1, padding=3),
            # act_func,
            nn.Conv1d(in_channels=input_channel*2, out_channels=input_channel*4, kernel_size=7, stride=2, padding=3),
            act_func,
            # nn.Conv1d(in_channels=input_channel*4, out_channels=input_channel*4, kernel_size=7, stride=1, padding=3),
            # act_func,
            nn.Conv1d(in_channels=input_channel*4, out_channels=input_channel*8, kernel_size=7, stride=2, padding=3),
            act_func,
        )
        self.z2_mean_layer = nn.Conv1d(in_channels=input_channel*8, out_channels=input_channel*8, kernel_size=1, stride=1)
        # self.z2_mean_layer = nn.Linear(in_features=8, out_features=8)
        self.z2_std_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_channel*8, out_channels=input_channel*8, kernel_size=1, stride=1),
            nn.Softplus(),
        )
        # self.z2_std_layer = nn.Sequential(
        #     nn.Linear(in_features=8, out_features=8),
        #     nn.Softplus(),
        # )
    
    def forward(self, input):
        h_x = self.encoder(input)
        qz2_mean = self.z2_mean_layer(h_x)
        qz2_std = self.z2_std_layer(h_x)
        qz2_std = torch.clip(qz2_std, min=clip_std_min, max=clip_std_max)
        return qz2_mean, qz2_std

class h_for_px(nn.Module):
    def __init__(self, input_channel, output_padding_list=[0, 0, 0]):
        super(h_for_px, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_channel*8, out_channels=input_channel*4, kernel_size=7, stride=2, padding=3, output_padding=output_padding_list[0]),
            act_func,
            # nn.ConvTranspose1d(in_channels=input_channel*4, out_channels=input_channel*4, kernel_size=7, stride=1, padding=3),
            # act_func,
            nn.ConvTranspose1d(in_channels=input_channel*4, out_channels=input_channel*2, kernel_size=7, stride=2, padding=3, output_padding=output_padding_list[1]),
            act_func,
            # nn.ConvTranspose1d(in_channels=input_channel*2, out_channels=input_channel*2, kernel_size=7, stride=1, padding=3),
            # act_func,
            nn.ConvTranspose1d(in_channels=input_channel*2, out_channels=input_channel, kernel_size=7, stride=2, padding=3, output_padding=output_padding_list[2]),
            act_func,
        )
    
    def forward(self, input):
        h_z = self.decoder(input)
        return h_z

class attention_rnn_net(nn.Module):
    def __init__(self, x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim,) -> None:
        super(attention_rnn_net, self).__init__()
        self.rnn = nn.GRU(input_size=x_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=False)
        # self.attention = nn.Sequential(
        #     nn.Linear(in_features=rnn_dim, out_features=rnn_dim),
        #     nn.Tanh(),
        #     nn.Linear(in_features=rnn_dim, out_features=window_size, bias=False),
        #     nn.Softmax(dim=1), 
        # )
        self.feature_extraction = nn.Sequential(
            nn.Linear(in_features=rnn_dim, out_features=rnn_dim),
            act_func,
            nn.Linear(in_features=rnn_dim, out_features=rnn_dim),
            act_func,
        )
    
    def forward(self, input: torch.Tensor):
        input_flip = torch.flip(input, dims=(-2, ))
        rnn_output_flip, _ = self.rnn(input_flip)
        rnn_output = torch.flip(rnn_output_flip, dims=(-2, ))
        # # print(f"rnn_output:{rnn_output.shape}")
        # attention_output = self.attention(rnn_output)
        # # print(f"attention_output:{attention_output.shape}")

        # M_t = torch.matmul(rnn_output.permute(0,2,1), attention_output)
        # # print(f"M_t:{M_t.shape}")
        # feature_out = self.feature_extraction(M_t.permute(0,2,1))
        feature_out = self.feature_extraction(rnn_output)
        return feature_out
        # return rnn_output

class q_net(nn.Module):
    def __init__(self, x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim, h_for_qz: nn.Module, h_for_px: nn.Module,) -> None:
        super(q_net, self).__init__()
        self.h_for_qz = h_for_qz
        self.h_for_px = h_for_px
        self.n_samples = n_samples
        self.rnn_dim = rnn_dim
        self.a_rnn = attention_rnn_net(x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim)
        self.z1_mean_layer = nn.Linear(in_features=rnn_dim+z1_dim, out_features=z1_dim)
        self.z1_std_layer = nn.Sequential(
            nn.Linear(in_features=rnn_dim+z1_dim, out_features=z1_dim),
            nn.Softplus()
        )
        self.z1_dim = z1_dim
        self.window_size = window_size
        self.flow_num = 1
        self.flows = nn.Sequential(*[PlanarNormalizingFlow(shape=(self.z1_dim,)) for _ in range(self.flow_num)])
    
    def forward(self, input: torch.Tensor, if_pretrain=False):
        if if_pretrain:
            qz_mean, qz_std = self.h_for_qz(input)
            # print(f"qnet qz_std:{torch.mean(qz_std)}")
            # print(f"qz_mean:{qz_mean.shape}")
            q_zx = Normal(loc=qz_mean, scale=qz_std)
            p_z = Normal(loc=torch.zeros_like(qz_mean).to(device=global_device), scale=torch.ones_like(qz_std).to(device=global_device))
            noise = p_z.sample((self.n_samples,))
            z2_sampled = noise.mul(qz_std) + qz_mean
            return z2_sampled, p_z, q_zx
        else:
            qz_mean, qz_std = self.h_for_qz(input)
            q_z2_dist = Normal(loc=qz_mean, scale=qz_std)
            p_z = Normal(loc=torch.zeros_like(qz_mean), scale=torch.ones_like(qz_std))
            noise = p_z.sample((self.n_samples,))
            z2_sampled = noise.mul(qz_std) + qz_mean
            z2_sampled_reshaped = z2_sampled.reshape(-1, z2_sampled.size()[-2], z2_sampled.size()[-1])
            # print(f"z2_sampled_reshaped:{z2_sampled_reshaped.shape}")
            h_z = self.h_for_px(z2_sampled_reshaped)
            h_z = h_z.permute(0, 2, 1)
            # print(f"h_z:{h_z.shape}")
            a_rnn_out = self.a_rnn(h_z)
            # print(f"a_rnn_out:{a_rnn_out.shape}")
            a_rnn_out = a_rnn_out.reshape(self.n_samples, -1, self.window_size, self.rnn_dim)
            q_z1_dist = RecurrentDistribution(
                input_q=a_rnn_out,
                mean_q_mlp=self.z1_mean_layer,
                std_q_mlp=self.z1_std_layer,
                z_dim=self.z1_dim,
                window_length=self.window_size
            )
            z1_sampled = q_z1_dist.sample(n_samples=(self.n_samples, input.size()[0], )).permute(1, 2, 0, 3)
            z1_sampled_flowed, z1_flow_log_det = self.flows((z1_sampled, torch.zeros(1).to(global_device))) # z^K 雅可比矩阵
            return z2_sampled, z1_sampled, z1_sampled_flowed, z1_flow_log_det, q_z2_dist, q_z1_dist

class p_net(nn.Module):
    def __init__(self, x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim, h_for_qz: nn.Module, h_for_px: nn.Module,) -> None:
        super(p_net, self).__init__()
        self.h_for_qz = h_for_qz
        self.h_for_px = h_for_px
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.pretrain_x_mean_layer = nn.Conv1d(in_channels=x_dim, out_channels=x_dim, kernel_size=1, stride=1)
        # self.pretrain_x_mean_layer = nn.Linear(in_features=window_size, out_features=window_size)
        self.pretrain_x_std_layer = nn.Sequential(
            nn.Conv1d(in_channels=x_dim, out_channels=x_dim, kernel_size=1, stride=1),
            nn.Softplus(),
        )
        # self.pretrain_x_std_layer = nn.Sequential(
        #     nn.Linear(in_features=window_size, out_features=window_size),
        #     nn.Softplus(),
        # )
        self.train_x_mean_layer = nn.Linear(in_features=rnn_dim, out_features=x_dim)
        self.train_x_std_layer = nn.Sequential(
            nn.Linear(in_features=rnn_dim, out_features=x_dim),
            nn.Softplus(),
        )
        self.z1_mean_layer = nn.Linear(in_features=x_dim+z1_dim, out_features=z1_dim)
        self.z1_std_layer = nn.Sequential(
            nn.Linear(in_features=x_dim+z1_dim, out_features=z1_dim),
            nn.Softplus()
        )
        self.z1_dim = z1_dim
        self.window_size = window_size
        self.flow_num = 20
        self.flows = nn.Sequential(*[PlanarNormalizingFlow(shape=(self.z1_dim,)) for _ in range(self.flow_num)])
        self.h_for_z1 = nn.Linear(in_features=z1_dim, out_features=x_dim)
        self.h_for_z = nn.Sequential(
            nn.Linear(in_features=x_dim+x_dim, out_features=rnn_dim),
            act_func,
            nn.Linear(in_features=rnn_dim, out_features=rnn_dim),
            act_func
        )

    def forward(self, sampled_z1, sampled_z2, if_pretrain=False):
        # input: sampled_z2
        if if_pretrain:
            h_z = self.h_for_px(sampled_z2)
            # print(f"h_z:{h_z.shape}")
    
            x_mean = self.pretrain_x_mean_layer(h_z).permute(0, 2, 1).reshape(self.n_samples, -1, self.window_size, self.x_dim)
            x_std = self.pretrain_x_std_layer(h_z).permute(0, 2, 1).reshape(self.n_samples, -1, self.window_size, self.x_dim)
            x_std = torch.clip(x_std, min=clip_std_min, max=clip_std_max)
            
            # print(f"pnet x_std:{torch.mean(x_std)}")
            p_xz_dist = Normal(loc=x_mean, scale=x_std)
            return p_xz_dist, x_mean, x_std
        else:
            pz2_dist = Normal(loc=torch.zeros_like(sampled_z2).to(device=global_device), scale=torch.ones_like(sampled_z2).to(device=global_device))
            sampled_z2_reshaped = sampled_z2.reshape(-1, sampled_z2.size()[-2], sampled_z2.size()[-1])
            h_z2 = self.h_for_px(sampled_z2_reshaped).permute(0, 2, 1).reshape(self.n_samples, -1, self.window_size, self.x_dim)
            # print(f"h_z2:{h_z2.shape}")
            pz1_dist = RecurrentDistribution(
                input_q=h_z2,
                mean_q_mlp=self.z1_mean_layer,
                std_q_mlp=self.z1_std_layer,
                z_dim=self.z1_dim,
                window_length=self.window_size
            )
            # z1_sampled = pz1_dist.sample(n_samples=(self.n_samples, input.size()[0], ))
            # z1_sampled_flowed, z1_flow_log_det = self.flows((z1_sampled, torch.zeros(1).to(global_device))) # z^K 雅可比矩阵
            h_z1 = self.h_for_z1(sampled_z1)
            # print(f"h_z1:{h_z1.shape}")
            h_z1_z2 = torch.concat((h_z1, h_z2), dim=-1)
            h_z = self.h_for_z(h_z1_z2)
            x_mean = self.train_x_mean_layer(h_z)
            x_std = self.train_x_std_layer(h_z)
            x_std = torch.clip(x_std, min=clip_std_min, max=clip_std_max)
            
            p_xz_dist = Normal(loc=x_mean, scale=x_std)
            return x_mean, x_std, pz2_dist, pz1_dist, p_xz_dist

class VAE(nn.Module):
    def __init__(self, x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim, output_padding_list) -> None:
        super(VAE, self).__init__()
        self.h_for_qz = h_for_qz(input_channel=x_dim)
        self.h_for_px = h_for_px(input_channel=x_dim, output_padding_list=output_padding_list)
        self.p_net = p_net(x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim, h_for_qz=self.h_for_qz, h_for_px=self.h_for_px)
        self.q_net = q_net(x_dim, z1_dim, z2_dim, window_size, n_samples, rnn_dim, dense_dim, h_for_qz=self.h_for_qz, h_for_px=self.h_for_px)
    
    def forward(self, input: torch.Tensor, if_pretrain=False):
        if if_pretrain:
        
            input_transpose = input.permute(0, 2, 1)
            # print(f"input_transpose:{input_transpose.shape}")
            z2_sampled, p_z, q_zx = self.q_net(input_transpose, if_pretrain=if_pretrain)
            z2_sampled_reshape = z2_sampled.reshape(-1, z2_sampled.size()[-2], z2_sampled.size()[-1])
            p_xz, x_mean, x_std = self.p_net(sampled_z1=None, sampled_z2=z2_sampled_reshape, if_pretrain=if_pretrain)
            return z2_sampled, p_z, q_zx, p_xz, x_mean, x_std
        else:
            input_transpose = input.permute(0, 2, 1)
            # print(f"input_transpose:{input_transpose.shape}")
            z2_sampled, z1_sampled, z1_sampled_flowed, z1_flow_log_det, q_z2_dist, q_z1_dist = self.q_net(input_transpose, if_pretrain=if_pretrain)
            # print(f"z1_sampled_flowed:{z1_sampled_flowed.shape} z2_sampled:{z2_sampled.shape}")
            x_mean, x_std, pz2_dist, pz1_dist, p_xz_dist = self.p_net(z1_sampled_flowed, z2_sampled, if_pretrain=if_pretrain)
            return z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std


