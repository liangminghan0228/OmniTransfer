import torch
import numpy as np
from torch import nn
from torch import optim

from omnianomaly.RecurrentDistribution import RecurrentDistribution
from omnianomaly.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal
from omnianomaly.lgssm import LinearGaussianStateSpaceModel
from data_config import *
from omnianomaly.vae import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
class OmniAnomaly:
    def __init__(self, x_dims: int,
                 z_dims: int, 
                 max_epochs: int, 
                 batch_size: int,
                 window_size: int,
                 learning_rate: float,
                 ):
        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = None
        # print(f"self._valid_step_freq:{self._valid_step_freq}")
        self._step = 0
        self._device = global_device
        self._vae = VAE_GRU(
            input_dim=self._x_dims,
            z_dim=self._z_dims,
            window_size=self._window_size,
            
        ).to(self._device)

        self._optimizer = optim.Adam(params=self._vae.parameters(), lr=learning_rate)
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self._optimizer, T_max=10, eta_min=0)

        self.loss =  {'train': [], 'valid': [], 'valid_recon': []}
        self.best_valid_loss = 9999999
        self.best_valid_recon = 9999999

    def freeze_layers(self, freeze_layer: str = None):
        if freeze_layer == 'rnn':
            freeze_a_layer(self._vae._encoder._rnn)
            freeze_a_layer(self._vae._decoder._rnn)
            freeze_a_layer(self._vae.flows)
        
        # elif freeze_layer == 'dense':
        #     freeze_a_layer(self._vae._encoder._linear)
        #     freeze_a_layer(self._vae._decoder._linear)
        #     freeze_a_layer(self._vae._encoder._mean)
        #     freeze_a_layer(self._vae._encoder._std)
        #     freeze_a_layer(self._vae._decoder._mean)
        #     freeze_a_layer(self._vae._decoder._std)
        
        # elif freeze_layer == 'pnf':
        #     freeze_a_layer(self._vae.flows)
        
        elif freeze_layer == 'seq':
            freeze_a_layer(self._vae._encoder._mean)
            freeze_a_layer(self._vae._encoder._std)
            self._vae.p_z.transition_matrix.requires_grad = False        
            self._vae.p_z.observation_matrix.requires_grad = False        
        else:
            print(f"no {freeze_layer}")

    def unfreeze_layers(self):
        unfreeze_a_layer(self._vae._encoder._rnn)
        unfreeze_a_layer(self._vae._decoder._rnn)
        unfreeze_a_layer(self._vae.flows)

    def init_layers(self, init_layer: str = None):
        
        if init_layer == 'dense':
            # self._vae._encoder._mean.reset_parameters()
            # self._vae._encoder._std[0].reset_parameters()
            self._vae._decoder._mean.reset_parameters()
            self._vae._decoder._std[0].reset_parameters()
            # self._vae._encoder._linear[0].reset_parameters()
            # self._vae._encoder._linear[2].reset_parameters()
            # self._vae._decoder._linear[0].reset_parameters()
            # self._vae._decoder._linear[2].reset_parameters()
        elif init_layer == 'std':
            # self._vae._encoder._mean.reset_parameters()
            # self._vae._encoder._std[0].reset_parameters()
            # self._vae._decoder._mean.reset_parameters()
            self._vae._decoder._std[0].reset_parameters()
            # self._vae._encoder._linear[0].reset_parameters()
            # self._vae._encoder._linear[2].reset_parameters()
            # self._vae._decoder._linear[0].reset_parameters()
            # self._vae._decoder._linear[2].reset_parameters()
        elif init_layer == 'pnf':
            for pnf_index in range(self._vae.flow_num):
                self._vae.flows[pnf_index].reset_parameters()
        elif init_layer == 'rnn':
            self._vae._encoder._rnn[0].reset_parameters()
            self._vae._decoder._rnn[0].reset_parameters()
        else:
            print(f"no {init_layer}")


    @staticmethod
    def ELBO_loss(x, z, z_flowed, q_zx: RecurrentDistribution, p_z: LinearGaussianStateSpaceModel, p_xz: Normal, flow_log_det: torch.Tensor,test_id=None):
        # if test_id is None:
        index_weight = index_loss_weight
        # else:
        #     index_weight = index_weight_by_use_center[test_id]

        # index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        # log_p_xz = torch.sum(p_xz.log_prob(x).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor), dim=-1)

        index_loss_weight_tensor = torch.tensor(index_weight).to(device=global_device)
        log_p_xz = torch.sum(p_xz.log_prob(x).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor), dim=-1)

        log_q_zx = torch.sum(q_zx.log_prob(z), dim=-1) - flow_log_det.sum(dim=-1)
        log_p_z = p_z.log_prob(z_flowed)
        # return loss, recon, kl
        return -torch.mean(log_p_xz+log_p_z-log_q_zx), -torch.mean(log_p_xz)
    
    def valid_log(self, valid_data, log_file, save_path, epoch,test_id=None):
        if_break = False
        for w in valid_data:
            w = w.to(self._device)
            q_zx, p_xz, z, z_sampled_flowed, x_mean, x_std, flow_log_det = None,None,None,None,None,None,None
            try:
                q_zx, p_xz, z, z_sampled_flowed, x_mean, x_std, flow_log_det = self._vae(w)
            except:
                if_break = True
                break                            
            
            valid_loss, valid_recon = OmniAnomaly.ELBO_loss(w, z, z_sampled_flowed, q_zx, self._vae.p_z, p_xz, flow_log_det,test_id)
            self.loss['valid_recon'][-1] += valid_recon.item()
            self.loss['valid'][-1] += valid_loss.item()
            # del valid_loss, valid_recon, q_zx, p_xz, z, x_mean, x_std, flow_log_det
            # torch.cuda.empty_cache()

        if self.loss['valid_recon'][-1] <= self.best_valid_recon and not if_break:
        # if self.loss['valid'][-1] <= self.best_valid_loss and not if_break:
            self.best_valid_loss = self.loss['valid'][-1]
            self.best_valid_recon = self.loss['valid_recon'][-1]

            self.save(save_path)
            print("***", end=' ')
            print("***", end=' ', file=log_file)
        
        print(f"epoch:{epoch}/{self._max_epochs} step:{self._step} valid_loss:{self.loss['valid'][-1]}  recon:{self.loss['valid_recon'][-1]} train_loss:{self.loss['train'][-1]}")
        print(f"epoch:{epoch}/{self._max_epochs} step:{self._step} valid_loss:{self.loss['valid'][-1]}  recon:{self.loss['valid_recon'][-1]} train_loss:{self.loss['train'][-1]}", file=log_file)

        return if_break

    def fit(self, values, save_path: pathlib.Path, valid_portion=0.3, scores=None, if_wrap=False, test_id=None):
        self._vae.is_train = True
        print(f"values[0].shape:{values[0].shape}")
        # if_drop_last = False if (dataset_type == 'data2' and training_period <= 2) else True
        if valid_portion > 0:
            train_values = values
            train_sliding_window = SlidingWindowDataLoader(
                SlidingWindowDataset(train_values, self._window_size, scores, if_wrap=if_wrap),
                batch_size=self._batch_size,
                shuffle=True,
                drop_last=False,
            )
            all_data = list(train_sliding_window)
            print(f"all_data:{len(all_data)} {all_data[0].shape}  {all_data[-1].shape}")
            if len(all_data) > 1:
                valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
                train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
                train_data = np.array(all_data)[train_index_list]
                valid_data = np.array(all_data)[valid_index_list]
            else:
                train_data = all_data
                valid_data = all_data
            # self._valid_step_freq = len(train_data)*(self._max_epochs) // 4 + 1
            # self._valid_step_freq = len(train_data) * 1 
            # print(f"len train:{len(train_data)} len valid:{len(valid_data)} _valid_step_freq:{self._valid_step_freq}")
        else:
            train_sliding_window = SlidingWindowDataLoader(
                SlidingWindowDataset(values, self._window_size),
                batch_size=self._batch_size,
                shuffle=True,
                drop_last=False
            )
        
        log_path = save_path.parent / 'log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')
        self.loss['train'].append(0)
        self.loss['valid'].append(0)
        self.loss['valid_recon'].append(0)
        self.valid_log(valid_data, log_file, save_path, 0,test_id)
        for epoch in range(1, self._max_epochs+1):
            if_break = False
            for i, w in enumerate(train_data):
                self.loss['train'].append(0)
                self.loss['valid'].append(0)
                self.loss['valid_recon'].append(0)
                
                w = w.to(self._device)
                self._optimizer.zero_grad()
                q_zx, p_xz, z, z_sampled_flowed, x_mean, x_std, flow_log_det = None,None,None,None,None,None,None
          
                q_zx, p_xz, z, z_sampled_flowed, x_mean, x_std, flow_log_det = self._vae(w)
          

                loss, recon = OmniAnomaly.ELBO_loss(w, z, z_sampled_flowed, q_zx, self._vae.p_z, p_xz, flow_log_det,test_id)

                # Backpropagation
                self._optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self._vae.parameters(), max_norm=10, norm_type=2)
                self._optimizer.step()
                self.loss['train'][-1] = loss.item()
                # del loss, q_zx, p_xz, z, x_mean, x_std, flow_log_det
                # torch.cuda.empty_cache()
                
                # valid and log
                # if self._step != 0 and self._step % self._valid_step_freq == 0 and valid_portion > 0:
                #     if_break = self.valid_log(valid_data, log_file, save_path, epoch)
                self._step += 1
            if epoch % global_valid_epoch_freq == 0:
                if_break = self.valid_log(valid_data, log_file, save_path, epoch,test_id)
            self.schedule.step()
            if self._step % learning_rate_decay_by_epoch == 0:
                for p in self._optimizer.param_groups:
                    p['lr'] *= learning_rate_decay_factor
            if if_break:
                break
   
        self.loss['valid'].append(0)
        self.loss['valid_recon'].append(0)
        self.valid_log(valid_data, log_file, save_path, epoch,test_id)
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
      
            q_zx, p_xz, z, z_sampled_flowed, x_mean, x_std, flow_log_det = self._vae(w)
  
                                
            recon_mean_list.append(torch.mean(x_mean[:, :, -1, :], dim=0).cpu().detach().numpy())
            recon_std_list.append(torch.mean(x_std[:, :, -1, :], dim=0).cpu().detach().numpy())
            score_list.append(torch.mean(p_xz.log_prob(w)[:, :, -1, :], dim=0).cpu().detach().numpy())
            z_list.append(torch.mean(z[:, :, -1, :], dim=0).cpu().detach().numpy())
        
        score = np.concatenate(score_list, axis=0)
        recon_mean = np.concatenate(recon_mean_list, axis=0)
        recon_std = np.concatenate(recon_std_list, axis=0)
        z = np.concatenate(z_list, axis=0)
        # print(f"score :{score.shape} recon_mean:{recon_mean.shape} z:{z.shape}")
        return score, recon_mean, recon_std, z


    def save(self, save_path: pathlib.Path):
        torch.save(self._vae.state_dict(), save_path)

    def restore(self, save_path: pathlib.Path):
        self._vae.load_state_dict(torch.load(save_path))