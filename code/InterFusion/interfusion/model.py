from textwrap import wrap
import torch
import numpy as np
from torch import nn
from torch import optim
from interfusion.planarNF import PlanarNormalizingFlow

from interfusion.RecurrentDistribution import RecurrentDistribution
from interfusion.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal
from interfusion.vae import VAE
from data_config import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True        
class InterFusion:
    def __init__(self, x_dims: int, z1_dims: int = 3, z2_dims: int = 3, max_epochs: int = 20, 
                 pre_max_epochs: int = 20, batch_size: int = 100, window_size: int = 60, n_samples: int = 10,
                 learning_rate: float = 1e-3, output_padding_list: list = [0,0,0]):
        self.x_dims = x_dims
        self.max_epochs = max_epochs
        self.pre_max_epochs = pre_max_epochs
        self.batch_size = batch_size
        self.valid_epoch_freq = global_valid_epoch_freq
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.window_size = window_size
        self.step = 0
        self.device = global_device
        self.n_samples = n_samples
        self.vae = VAE(x_dims, z1_dims, z2_dims, window_size, n_samples, global_rnn_dims, global_dense_dims, output_padding_list).to(device=global_device)

        self.optimizer = optim.Adam(params=self.vae.parameters(), lr=learning_rate)
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
        self.pretrain_loss =  {'train': [], 'valid': []}
        self.train_loss =  {'train': [], 'valid': [], 'valid_recon': [], 'valid_kl': []}
        self.best_pretrain_valid_loss = 9999999
        self.best_train_valid_loss = 9999999
        self.best_train_valid_recon = 9999999
    
    def freeze_layers(self, name):
        if name == 'rnn':
            freeze_a_layer(self.vae.q_net.a_rnn.rnn)
        elif name == 'cnn':
            freeze_a_layer(self.vae.h_for_qz)
            freeze_a_layer(self.vae.h_for_px)
    def unfreeze_layers(self):
        unfreeze_a_layer(self.vae.q_net.a_rnn.rnn)
        unfreeze_a_layer(self.vae.h_for_qz)
        unfreeze_a_layer(self.vae.h_for_px)
    
    def init_layers(self, name):
        if name == 'std':
            # self.vae.p_net.train_x_mean_layer.reset_parameters()
            self.vae.p_net.train_x_std_layer[0].reset_parameters()
        elif name == 'dense':
            self.vae.p_net.train_x_mean_layer.reset_parameters()
            self.vae.p_net.train_x_std_layer[0].reset_parameters()            
        # self.vae.q_net.z1_mean_layer.reset_parameters()
        # self.vae.q_net.z1_std_layer[0].reset_parameters()
        # self.vae.q_net.a_rnn.feature_extraction[0].reset_parameters()
        # self.vae.q_net.a_rnn.feature_extraction[2].reset_parameters()
        # self.vae.p_net.z1_mean_layer.reset_parameters()
        # self.vae.p_net.h_for_z1.reset_parameters()
        # self.vae.p_net.z1_std_layer[0].reset_parameters()
        # self.vae.p_net.h_for_z[0].reset_parameters()
        # self.vae.p_net.h_for_z[2].reset_parameters()
            
    @staticmethod
    def pretrain_ELBO_loss(x, z, q_zx: RecurrentDistribution, p_z: Normal, p_xz: Normal):
        # 预训练
        # print(f"pretrain_ELBO_loss x:{x.shape} z:{z.shape}")
        index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        log_p_xz = torch.sum(p_xz.log_prob(x).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor), dim=(-2, -1)) # samplen, batch, time, xdim
        log_q_zx = torch.sum(q_zx.log_prob(z), dim=(-2, -1))  # samplen, batch, xdim, time
        log_p_z = torch.sum(p_z.log_prob(z), dim=(-2, -1))  # samplen, batch, xdim, time
        # print(f"pretrain_ELBO_loss log_p_xz:{log_p_xz.shape} log_q_zx:{log_q_zx.shape} log_p_z:{log_p_z.shape}")
        # print(f"pretrain_ELBO_loss log_p_xz:{-torch.mean(log_p_xz)} log_q_zx:{-torch.mean(log_q_zx)} log_p_z:{-torch.mean(log_p_z)}")
        return -torch.mean(log_p_xz+log_p_z-log_q_zx)
    
    @staticmethod
    def train_ELBO_loss(x: torch.Tensor, z1_sampled: torch.Tensor, z1_sampled_flowed: torch.Tensor, z1_flow_log_det: torch.Tensor, z2_sampled: torch.Tensor, 
        p_xz_dist: Normal, q_z1_dist: RecurrentDistribution, q_z2_dist: Normal, pz1_dist: RecurrentDistribution, pz2_dist: Normal):
        # print(f"train_ELBO_loss---")
        # print(f"z1_flow_log_det {z1_flow_log_det.shape}")
        index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        log_p_xz = torch.sum(p_xz_dist.log_prob(x).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor), dim=(-2, -1))
        # print(f" log_p_xz:{log_p_xz.shape}")
        log_p_z1 = torch.sum(pz1_dist.log_prob(z1_sampled_flowed), dim=(-2, -1))
        # print(f"log_p_z1:{log_p_z1.shape}")
        log_p_z2 = torch.sum(pz2_dist.log_prob(z2_sampled), dim=(-2, -1))
        # print(f"log_p_z2:{log_p_z2.shape}")
        log_p_z = log_p_z1 + log_p_z2
        # print(f"q_z1_dist:{q_z1_dist.log_prob(z1_sampled).shape}")
        log_q_z1 = torch.sum(
            torch.sum(q_z1_dist.log_prob(z1_sampled), dim=(-1,)) - 
            torch.sum(z1_flow_log_det, dim=-1), 
            dim=(-1,))
        log_q_z2 = torch.sum(q_z2_dist.log_prob(z2_sampled), dim=(-2, -1))
        # print(f"log_q_z1:{log_q_z1.shape}")
        # print(f"log_q_z2:{log_q_z2.shape}")
        log_q_zx = log_q_z1 + log_q_z2
        # print(f"train_ELBO_loss log_p_xz:{-torch.mean(log_p_xz)} log_q_zx:{-torch.mean(log_q_zx)} log_p_z:{-torch.mean(log_p_z)}")
        
        return -torch.mean(log_p_xz+log_p_z-log_q_zx), -torch.mean(log_p_xz), -torch.mean(log_p_z-log_q_zx)
    
    def valid_log_pretrain(self, valid_data, log_file, save_path, epoch):
        # train_type='train' 'pretrain'
        if_break = False
        for w in valid_data:
            w = w.to(global_device)
            q_zx, p_xz, z, x_mean, x_std = None,None,None,None,None
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
            # try:
            z2_sampled, p_z, q_zx, p_xz, x_mean, x_std = self.vae(w, if_pretrain=True)
            valid_loss = InterFusion.pretrain_ELBO_loss(w, z2_sampled, q_zx, p_z, p_xz)
            self.pretrain_loss['valid'][-1] += valid_loss.item()

        # del q_zx, p_xz, z, x_mean, x_std
        if self.pretrain_loss['valid'][-1] <= self.best_pretrain_valid_loss and not if_break:
            self.best_pretrain_valid_loss = self.pretrain_loss['valid'][-1]
            if self.best_pretrain_valid_loss < -1000:
        
                print(f"valid loss low:{self.pretrain_loss['valid'][-1]}")
                print(f"valid loss low:{self.pretrain_loss['valid'][-1]}", file=log_file)
                if_break = True
            self.save(save_path)
            print("***", end=' ')
            print("***", end=' ', file=log_file)    
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.pretrain_loss['valid'][-1]} train_loss:{self.pretrain_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}")
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.pretrain_loss['valid'][-1]} train_loss:{self.pretrain_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}", file=log_file)
        
        return if_break

    def valid_log_train(self, valid_data, log_file, save_path, epoch):
        # train_type='train' 'pretrain'
        if_break = False
        for w in valid_data:
            w = w.to(global_device)
            q_zx, p_xz, z, x_mean, x_std = None,None,None,None,None
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
            # try:
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)
            valid_loss, valid_recon, valid_kl = InterFusion.train_ELBO_loss(x=w, z1_sampled=z1_sampled, z1_sampled_flowed=z1_sampled_flowed, z1_flow_log_det=z1_flow_log_det, z2_sampled=z2_sampled, 
                p_xz_dist=p_xz_dist, q_z1_dist=q_z1_dist, q_z2_dist=q_z2_dist, pz1_dist=pz1_dist, pz2_dist=pz2_dist)
            self.train_loss['valid'][-1] += valid_loss.item()
            self.train_loss['valid_recon'][-1] += valid_recon.item()
            self.train_loss['valid_kl'][-1] += valid_kl.item()
           
        # del z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std
        if self.train_loss['valid_recon'][-1] <= self.best_train_valid_recon and not if_break:
            self.best_train_valid_loss = self.train_loss['valid'][-1]
            self.best_train_valid_recon = self.train_loss['valid_recon'][-1]

            self.save(save_path)   
            print("***", end=' ')
            print("***", end=' ', file=log_file)                                           
        if self.train_loss['valid'][-1] < -100000:
            print(f"valid loss low:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}")
            print(f"valid loss low:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}", file=log_file)
            return True
        
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}")
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}", file=log_file)

        
        return False
    

    def get_data(self, values, valid_portion):
        print(values[0].shape)
        if_drop_last = False if (dataset_type == 'data2' and training_period <= 2) else True
        train_values = values
        print(train_values[0].shape)
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self.window_size, if_wrap=True),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=if_drop_last
        )
        all_data = list(train_sliding_window)
        print(f"all_data:{len(all_data)} {all_data[0].shape}")
        if len(all_data) > 1:
            valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
            train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
            train_data = np.array(all_data)[train_index_list]
            valid_data = np.array(all_data)[valid_index_list]
        else:
            train_data = all_data
            valid_data = all_data

        return train_data, valid_data

    def prefit(self, values, save_path: pathlib.Path, valid_portion=0.3):
        train_data, valid_data = self.get_data(values=values, valid_portion=valid_portion)
        log_path = save_path.parent / 'pretrain_log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')

        for epoch in range(1, self.pre_max_epochs+1):
            if_break = False
            for i, w in enumerate(train_data):
                self.pretrain_loss['train'].append(0)
                self.pretrain_loss['valid'].append(0)
                w = w.to(global_device)
                self.optimizer.zero_grad()
                z2_sampled, p_z, q_zx, p_xz = None,None,None,None
    
                z2_sampled, p_z, q_zx, p_xz, x_mean, x_std = self.vae(w, if_pretrain=True)
  
                loss = InterFusion.pretrain_ELBO_loss(w, z2_sampled, q_zx, p_z, p_xz)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.pretrain_loss['train'][-1] = loss.item()
                self.step += 1

            if epoch % self.valid_epoch_freq == 0 and valid_portion > 0:
                if_break = self.valid_log_pretrain(valid_data, log_file, save_path, epoch)
            if epoch % learning_rate_decay_by_epoch == 0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= learning_rate_decay_factor
            if dataset_type != 'ctf':
                self.schedule.step()
            if if_break:
                break
     
        self.valid_log_pretrain(valid_data, log_file, save_path, epoch)


    def fit(self, values, save_path: pathlib.Path, valid_portion=0.3):
        train_data, valid_data = self.get_data(values=values, valid_portion=valid_portion)
        log_path = save_path.parent / 'train_log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')

        self.train_loss['train'].append(0)  
        self.train_loss['valid'].append(0)
        self.train_loss['valid_recon'].append(0)
        self.train_loss['valid_kl'].append(0)
        self.valid_log_train(valid_data, log_file, save_path, 0)

        for epoch in range(1, self.max_epochs+1):
            if_break = False
            for i, w in enumerate(train_data):
                self.train_loss['train'].append(0)
                self.train_loss['valid'].append(0)
                self.train_loss['valid_recon'].append(0)
                self.train_loss['valid_kl'].append(0)
                
                w = w.to(global_device)
                self.optimizer.zero_grad()
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
           
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)
     
                loss, _, _ = InterFusion.train_ELBO_loss(x=w, z1_sampled=z1_sampled, z1_sampled_flowed=z1_sampled_flowed, z1_flow_log_det=z1_flow_log_det, z2_sampled=z2_sampled, 
                    p_xz_dist=p_xz_dist, q_z1_dist=q_z1_dist, q_z2_dist=q_z2_dist, pz1_dist=pz1_dist, pz2_dist=pz2_dist)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.train_loss['train'][-1] = loss.item()
                # del z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std, _
                # torch.cuda.empty_cache()
                # valid and log

                self.step += 1
            if epoch % self.valid_epoch_freq == 0 and valid_portion > 0:
                if_break = self.valid_log_train(valid_data, log_file, save_path, epoch)
                if if_break:
                    break
            if epoch % learning_rate_decay_by_epoch == 0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= learning_rate_decay_factor
            if dataset_type != 'ctf':
                self.schedule.step()
    
        self.valid_log_train(valid_data, log_file, save_path, epoch)

    def predict(self, values, save_path: pathlib.Path, if_pretrain=False):
        self.vae.eval()
        score_list = []
        recon_mean_list = []
        recon_std_list = []
        z_list = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self.window_size),
            batch_size=self.batch_size,
        )
        log_path = save_path.parent / 'log.txt'
        log_file = open(log_path, mode='a')
        for w in test_sliding_window:
            w = w.to(self.device)
            # try:
            if if_pretrain:
                z2_sampled, p_z, q_zx, p_xz_dist, x_mean, x_std = self.vae(w, if_pretrain=True)
                z_list.append(torch.mean(z2_sampled[:, :, :,-1], dim=0).cpu().detach().numpy())
            else:
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)
                z_list.append(torch.mean(z1_sampled_flowed[:, :, -1, :], dim=0).cpu().detach().numpy())
             
            recon_mean_list.append(torch.mean(x_mean[:, :, -1, :], dim=0).cpu().detach().numpy())
            recon_std_list.append(torch.mean(x_std[:, :, -1, :], dim=0).cpu().detach().numpy())
            score_list.append(torch.mean(p_xz_dist.log_prob(w)[:, :, -1, :], dim=0).cpu().detach().numpy())
        
        score = np.concatenate(score_list, axis=0)
        recon_mean = np.concatenate(recon_mean_list, axis=0)
        recon_std = np.concatenate(recon_std_list, axis=0)
        z = np.concatenate(z_list, axis=0)
        # print(f"score :{score.shape} recon_mean:{recon_mean.shape} z:{z.shape}")
        return score, recon_mean, recon_std, z


    def save(self, save_path: pathlib.Path):
        torch.save(self.vae.state_dict(), save_path)

    def restore(self, save_path: pathlib.Path):
        self.vae.load_state_dict(torch.load(save_path))