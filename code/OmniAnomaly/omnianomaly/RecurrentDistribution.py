import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal
from data_config import *




class RecurrentDistribution(Distribution):
    def __init__(self, input_q: torch.Tensor, mean_q_mlp, std_q_mlp, z_dim, window_length):
        super(RecurrentDistribution, self).__init__(validate_args=False)
        self.std_q_mlp = std_q_mlp
        self.mean_q_mlp = mean_q_mlp
        self.input_q = input_q.permute(1, 0, 2) # window_length * n_samples * z_dim
        self.z_dim = z_dim
        self.window_length = window_length

    def sample(self, n_samples):
        noise = torch.empty(n_samples + (self.window_length, self.z_dim)).to(device=global_device)
        nn.init.trunc_normal_(noise, a=-2, b=2)
        noise = noise.permute(2, 0, 1, 3) # window_length * n_samples * z_dim
        time_indices_shape = n_samples + (self.z_dim,)
        # print(f"time_indices_shape:{time_indices_shape}")
        state = (torch.zeros(time_indices_shape).to(device=global_device), torch.zeros(time_indices_shape).to(device=global_device), torch.ones(time_indices_shape).to(device=global_device))
        z_sampled = torch.zeros(noise.size()).to(device=global_device)
        # with torch.no_grad():
        for time_step in range(self.window_length):
            z_previous, mu_q_previous, std_q_previous = state
            noise_n, input_q_n = noise[time_step], self.input_q[time_step]
            input_q_n = input_q_n.broadcast_to(z_previous.size()[:-1]+input_q_n.size()[-1:])
            input_q_cat = torch.concat([input_q_n, z_previous], dim=-1)
            # print(f"input_q_cat:{input_q_cat.size()}")
            mu_q = self.mean_q_mlp(input_q_cat)
            std_q = self.std_q_mlp(input_q_cat)
            std_q = torch.clamp(std_q, min=min_std)
            # print(f"noise_n:{noise_n.size()} std_q:{std_q.size()}")
            temp = torch.mul(noise_n, std_q)
            mu_q = mu_q.broadcast_to(temp.size())
            std_q = std_q.broadcast_to(temp.size())
            z_n = temp + mu_q
            z_sampled[time_step] = z_n
            state = (z_n, mu_q, std_q)

        # print(f"z_sampled:{z_sampled.shape} q_std:{q_std.shape} q_mu:{q_mu.shape}")
        return z_sampled

    def log_prob(self, given: torch.Tensor):
        given = given.permute(2, 0, 1, 3) # windows_size, n_sample, batch, z_dim
        log_prob_res = torch.zeros(given.size()).to(device=global_device)
        # with torch.no_grad():
        
        for time_step in range(self.window_length):
            given_n, input_q_n = given[time_step], self.input_q[time_step]
            input_q_n = input_q_n.broadcast_to(given_n.size()[:-1]+input_q_n.size()[-1:])
            input_q_concat = torch.concat([input_q_n, given_n], dim=-1)
            mu_q = self.mean_q_mlp(input_q_concat)
            std_q = self.std_q_mlp(input_q_concat)
            std_q = torch.clamp(std_q, min=min_std)
            log_std_q = torch.log(std_q)
            precision = torch.exp(-2 * log_std_q)
            log_prob_n = - 0.9189385332046727 - log_std_q - 0.5 * precision * torch.square(
                torch.minimum(
                    torch.abs(given_n - mu_q), torch.tensor(1e8)
                    ))
            log_prob_res[time_step] = log_prob_n

        
        log_prob_res = log_prob_res.permute(1, 2, 0, 3)
        return log_prob_res

            






