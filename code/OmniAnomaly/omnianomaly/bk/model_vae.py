import torch
from torch import nn
from torch import optim

from omnianomaly.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal
from data_config import *



class AE_layer(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_dim, dense_dim):
        print('AE_layer', input_dim, output_dim, rnn_dim, dense_dim)
        super(AE_layer, self).__init__()

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
        
    
    def forward(self, x):
        e_output, e_hidden = self._rnn(x)
        # print(f"output:{e_output.shape} hidden:{e_hidden.shape}")
        e_output = self._linear(e_output)
        _mean = self._mean(e_output)
        _std = self._std(e_output)
        return _mean, _std

class VAE_GRU(nn.Module):
    def __init__(self, encoder, decoder, n_sample=10):
        super(VAE_GRU, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._n_sample = n_sample
        self.is_train = True

    def forward(self, x):
        z_mean, z_std = self._encoder(x)
        # print(f"z_mean:{torch.mean(z_mean)} z_std:{torch.mean(z_std)}")
        q_zx = Normal(z_mean, z_std)
        p_z = Normal(
            torch.zeros_like(z_mean).to(device=global_device),
            torch.ones_like(z_std).to(device=global_device)
        )
        noise = p_z.sample((self._n_sample,)) * torch.unsqueeze(z_std, 0) + torch.unsqueeze(z_mean, 0)
        z = noise.mul(z_std) + z_mean

        z_reshape = z.view(-1, z.shape[-2], z.shape[-1])
        x_mean, x_std = self._decoder(z_reshape)
        # print(f"x_mean:{x_mean.shape}")
        x_mean_reshape = x_mean.view(self._n_sample, x.shape[0], x_mean.shape[-2], x_mean.shape[-1])
        x_std_reshape = x_std.view(self._n_sample, x.shape[0], x_std.shape[-2], x_std.shape[-1])
        p_xz = Normal(x_mean_reshape, x_std_reshape)
        return p_z, q_zx, p_xz, z, x_mean_reshape, x_std_reshape


class OmniAnomaly:
    def __init__(self, x_dims: int,
                 z_dims: int = 3, 
                 rnn_dims: int = 100, 
                 dense_dims: int = 100, 
                 max_epochs: int = 250, 
                 batch_size: int = 128,
                 window_size: int = 60,
                 valid_step_frep: int = 5,
                 learning_rate: float = 1e-3,
                 ):
        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = valid_step_frep
        print(f"self._valid_step_freq:{self._valid_step_freq}")
        self._step = 0
        self._device = torch.device('cuda')
        self._vae = VAE_GRU(
            AE_layer(x_dims, z_dims, rnn_dims, dense_dims), 
            AE_layer(z_dims, x_dims, rnn_dims, dense_dims)
        ).to(self._device)

        self._optimizer = optim.Adam(params=self._vae.parameters(), lr=learning_rate)

        self.loss =  {'train': [0], 'valid': [0]}
    
    @staticmethod
    def ELBO_loss(x, z, q_zx: Normal, p_z: Normal, p_xz: Normal):

        log_p_xz = torch.sum(p_xz.log_prob(x), dim=-1)
        # print(f"p_xz.log_prob(x): {p_xz.log_prob(x).shape}")
        log_q_zx = torch.sum(q_zx.log_prob(z), dim=-1)
        # print(f"p_z.log_prob(z): {p_z.log_prob(z).shape}")
        log_p_z = torch.sum(p_z.log_prob(z), dim=-1)
        # print(f"log_p_z: {log_p_z.shape}")
        # print(f"torch.mean(log_p_xz):{torch.mean(log_p_xz)} torch.mean(log_q_zx):{torch.mean(log_q_zx)} torch.mean(log_p_z):{torch.mean(log_p_z)}")
        return -torch.mean(log_p_xz+log_p_z-log_q_zx)

    def fit(self, values, save_path, valid_portion=0.3):
        self._vae.is_train = True
        valid_len = int(len(values[0]) * valid_portion)
        # train_values, valid_values = values[:-valid_len], values[-valid_len:]
        train_values, valid_values = [v[:-valid_len] for v in values], [v[-valid_len:] for v in values]
        print(train_values[0].shape)
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True
        )

        valid_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(valid_values, self._window_size),
            batch_size=self._batch_size,
        )
        best_valid_loss = 9999999
        log_path = save_path.parent / 'log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')

        for epoch in range(1, self._max_epochs+1):
            for i, w in enumerate(train_sliding_window):
                w = w.to(self._device)
                self._optimizer.zero_grad()
                p_z, q_zx, p_xz, z, x_mean, x_std = self._vae(w)

                loss = OmniAnomaly.ELBO_loss(w, z, q_zx, p_z, p_xz)

                # Backpropagation
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self.loss['train'][-1] = loss.item()

                # valid and log
                if self._step != 0 and self._step % self._valid_step_freq == 0:
                    for w in valid_sliding_window:
                        w = w.to(self._device)
                        p_z, q_zx, p_xz, z, x_mean, x_std = self._vae(w)
                        valid_loss = OmniAnomaly.ELBO_loss(w, z, q_zx, p_z, p_xz)
                        self.loss['valid'][-1] += valid_loss.item()
                    
                    if self.loss['valid'][-1] < best_valid_loss:
                        best_valid_loss = self.loss['valid'][-1]
                        # 保存最佳模型
                        self.save(save_path)
                        print("***", end=' ')
                        print("***", end=' ', file=log_file)
                    print(f"epoch:{epoch}/{self._max_epochs} step:{self._step} valid_loss:{self.loss['valid'][-1]} train_loss:{self.loss['train'][-1]}")
                self.loss['train'].append(0)
                self.loss['valid'].append(0)
                self._step += 1

    def predict(self, values, save_path: pathlib.Path):
        self._vae.eval()
        self._vae.is_train = False
        score_list = []
        recon_mean_list = []
        recon_std_list = []
        z_list = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size),
            batch_size=self._batch_size,
        )
        log_path = save_path.parent / 'log.txt'
        log_file = open(log_path, mode='a')
        for w in test_sliding_window:
            w = w.to(self._device)
            p_z, q_zx, p_xz, z, x_mean, x_std = self._vae(w)
  
            recon_mean_list.append(torch.mean(x_mean[:, :, -1, :], dim=0).cpu().detach().numpy())
            recon_std_list.append(torch.mean(x_std[:, :, -1, :], dim=0).cpu().detach().numpy())
            score_list.append(torch.mean(p_xz.log_prob(w)[:, :, -1, :], dim=0).cpu().detach().numpy())
            z_list.append(torch.mean(z[:, :, -1, :], dim=0).cpu().detach().numpy())
        score = np.concatenate(score_list, axis=0)
        recon_mean = np.concatenate(recon_mean_list, axis=0)
        recon_std = np.concatenate(recon_std_list, axis=0)
        z = np.concatenate(z_list, axis=0)
        print(f"score :{score.shape} recon_mean:{recon_mean.shape} z:{z.shape}")
        return score, recon_mean, recon_std, z


    def save(self, save_path):
        torch.save(self._vae.state_dict(), save_path)

    def restore(self, save_path):
        self._vae.load_state_dict(torch.load(save_path))
