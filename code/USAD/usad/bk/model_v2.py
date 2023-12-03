
import torch
import numpy as np
from torch import nn
from torch import optim

from typing import List, Sequence

from usad.data import SlidingWindowDataset, SlidingWindowDataLoader

import time

from data_config import *

def freeze(layer, freeze_index_list):
    for layer_index, child in enumerate(layer.children()):
        if layer_index in freeze_index_list:
            
            for param in child.parameters():
                param.requires_grad = False


class Encoder(nn.Module):

    def __init__(self, input_dims: int, z_dims: int, nn_size: Sequence[int] = None, freeze_index_list: List =[]):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)
        layers = []
        last_size = input_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            # layers.append(nn.ReLU())
            # layers.append(nn.Sigmoid())
            layers.append(nn.ELU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, z_dims))
        # layers.append(nn.Tanh())
        # layers.append(nn.Sigmoid())
        # layers.append(nn.ELU())
        self._net = nn.Sequential(*layers)
        freeze(self._net, freeze_index_list)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        z = self._net(w)
        return z


class Decoder(nn.Module):

    def __init__(self, z_dims: int, input_dims: int, nn_size: Sequence[int] = None, freeze_index_list: List =[]):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)

        layers = []
        last_size = z_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            # layers.append(nn.ReLU())
            # layers.append(nn.Sigmoid())
            layers.append(nn.ELU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, input_dims))
        # layers.append(nn.Tanh())
        # layers.append(nn.ELU())
        self._net = nn.Sequential(*layers)
        freeze(self._net, freeze_index_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self._net(z)
        return w


class USAD:

    def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size: Sequence[int] = None, decoder_nn_size: Sequence[int] = None,
                 z_dims: int = 38, window_size: int = 10, valid_step_frep: int = 200, freeze_index_list_encoder: List = [], freeze_index_list_decoder: List = [], require_improvement: int = 10, global_lr: float = 1e-3):
        self._x_dims = x_dims
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = valid_step_frep
        self._step = 0
        self.require_improvement = require_improvement 
        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims, freeze_index_list=freeze_index_list_encoder).to(global_device)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims, freeze_index_list=freeze_index_list_decoder).to(global_device)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims, freeze_index_list=freeze_index_list_decoder).to(global_device)
        self.loss_func = nn.MSELoss(reduction='none')
        self._optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, list(self._shared_encoder.parameters()) + list(self._decoder_G.parameters()) ), lr=global_lr)
        self._optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, list(self._shared_encoder.parameters()) + list(self._decoder_D.parameters()) ), lr=global_lr)
        self.loss = {'AE_G': {'train': [0], 'valid': [0]},
                     'AE_D': {'train': [0], 'valid': [0]}}
        self.valid_value = [0]
    
    def compose_loss(self, w, w_G, w_G_D, w_D, epoch, loss_type='train'):
        index_loss_weight_tensor = torch.tensor(index_loss_weight).cuda()
        mse_left_G = self.loss_func(w_G, w)
        mse_left_G = mse_left_G.view(mse_left_G.size()[0], self._window_size, self._x_dims)
        mse_left_G = torch.mean(mse_left_G.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
        
        mse_right_G = self.loss_func(w_G_D, w)
        mse_right_G = mse_right_G.view(mse_right_G.size()[0], self._window_size, self._x_dims)
        mse_right_G = torch.mean(mse_right_G.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
        loss_G = (1 / epoch) * mse_left_G + (1 - 1 / epoch) * mse_right_G

        self.loss['AE_G'][loss_type][-1] += loss_G.item()

        mse_left_D = self.loss_func(w_D, w)
        mse_left_D = mse_left_D.view(mse_left_D.size()[0], self._window_size, self._x_dims)
        mse_left_D = torch.mean(mse_left_D.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
        mse_right_D = self.loss_func(w_G_D, w)
        mse_right_D = mse_right_D.view(mse_right_D.size()[0], self._window_size, self._x_dims)
        mse_right_D = torch.mean(mse_right_D.mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor))
        loss_D = (1 / epoch) * mse_left_D - (1 - 1 / epoch) * mse_right_D

        self.loss['AE_D'][loss_type][-1] += loss_D.item()

        return loss_G, loss_D

    def fit(self, values, model_save_dir: pathlib.Path, valid_portion=0.3, local_epoch=None):
        # valid_len = int(len(values[0]) * valid_portion)
        # train_values, valid_values = [v[:-valid_len] for v in values], [v[-valid_len:] for v in values]
        train_values = values
        if_drop_last = False if (dataset_type == 'data2' and training_period <= 2) else True
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=if_drop_last
        )
        # valid_sliding_window = SlidingWindowDataLoader(
        #     SlidingWindowDataset(valid_values, self._window_size),
        #     batch_size=self._batch_size,
        # )
        all_data = list(train_sliding_window)
        valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
        train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
        train_data = np.array(all_data)[train_index_list]
        valid_data = np.array(all_data)[valid_index_list]

        total_time = 0
        log_path = model_save_dir / f"log.txt"
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')
        max_delta_loss_gd = 0
        min_wgd_w_delta = 9999999999
        for epoch in range(1, local_epoch+1):
            st_epoch = time.time()

            for i, w in enumerate(train_data):
                # print(self._step, w.shape)
                w = w.view(-1, self._input_dims)
                w = w.to(global_device)

                self._optimizer_G.zero_grad()
                self._optimizer_D.zero_grad()

                z = self._shared_encoder(w)
                w_G = self._decoder_G(z)
                w_D = self._decoder_D(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G))

                loss_G, loss_D = self.compose_loss(w, w_G, w_G_D, w_D, epoch)
                # backward
                loss_G.backward(retain_graph=True)
                loss_D.backward()

                self._optimizer_G.step()
                self._optimizer_D.step()

                # valid and log
                if self._step > 0 and self._step % self._valid_step_freq == 0:
                    for w in valid_data:
                        w = w.view(-1, self._input_dims)
                        w = w.to(global_device)
                        z = self._shared_encoder(w)
                        w_G = self._decoder_G(z)
                        w_G_D = self._decoder_D(self._shared_encoder(w_G))
                        self.valid_value[-1] += torch.sum(self.loss_func(w, w_G_D)).cpu().detach().item()
                    if self.valid_value[-1] < min_wgd_w_delta:
                        min_wgd_w_delta = self.valid_value[-1]
                        self.save(model_save_dir)
                        print(f"***", end=' ')
                        print(f"***", end=' ', file=log_file)
                    print(f"epoch:{epoch} step:{self._step} train loss_G:{self.loss['AE_G']['train'][-1]} loss_D:{self.loss['AE_D']['train'][-1]} valid w-wgd:{self.valid_value[-1]}")
                    print(f"epoch:{epoch} step:{self._step} train loss_G:{self.loss['AE_G']['train'][-1]} loss_D:{self.loss['AE_D']['train'][-1]} valid w-wgd:{self.valid_value[-1]}", file=log_file)

                    self.loss['AE_G']['train'].append(0)
                    self.loss['AE_D']['train'].append(0)
                    self.loss['AE_G']['valid'].append(0)
                    self.loss['AE_D']['valid'].append(0)
                    self.valid_value.append(0)

                self._step += 1
                if self._step % learning_rate_decay_by_step == 0:
                    for p in self._optimizer_G.param_groups:
                        p['lr'] *= learning_rate_decay_factor
                    for p in self._optimizer_D.param_groups:
                        p['lr'] *= learning_rate_decay_factor
            et_epoch = time.time()
            total_time += (et_epoch - st_epoch)
            
    def predict(self, values):
        collect_scores_d = []
        collect_scores_g = []
        w_G_list = []
        w_G_D_list = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size),
            batch_size=self._batch_size,
        )
        with torch.no_grad():
            for w in test_sliding_window:
                w = w.view(-1, self._input_dims)
                w = w.to(global_device)

                z = self._shared_encoder(w)
                w_G = self._decoder_G(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G))
                batch_scores_d = self.loss_func(w_G, w)
                batch_scores_g = self.loss_func(w_G_D, w)
                batch_scores_d = batch_scores_d.view(-1, self._window_size, self._x_dims)
                batch_scores_g = batch_scores_g.view(-1, self._window_size, self._x_dims)
                batch_scores_d = batch_scores_d.cuda().data.cpu().numpy() 
                batch_scores_g = batch_scores_g.cuda().data.cpu().numpy()
                w_G = w_G.view(-1, self._window_size, self._x_dims).cpu().detach().numpy()
                w_G_D = w_G_D.view(-1, self._window_size, self._x_dims).cpu().detach().numpy()
                if not collect_scores_d:
                    collect_scores_d.extend(batch_scores_d[0])
                    collect_scores_g.extend(batch_scores_g[0])
                    collect_scores_d.extend(batch_scores_d[1:, -1])
                    collect_scores_g.extend(batch_scores_g[1:, -1])
                    w_G_list.extend(w_G[0])
                    w_G_list.extend(w_G[1:, -1])
                    w_G_D_list.extend(w_G_D[0])
                    w_G_D_list.extend(w_G_D[1:, -1])
                else:
                    collect_scores_d.extend(batch_scores_d[:, -1])
                    collect_scores_g.extend(batch_scores_g[:, -1])
                    w_G_list.extend(w_G[:, -1])
                    w_G_D_list.extend(w_G_D[:, -1])

        return collect_scores_d, collect_scores_g, w_G_list, w_G_D_list

    def save(self, model_save_dir):
        shared_encoder_path = model_save_dir / 'shared_encoder.pkl'
        decoder_G_path = model_save_dir / 'decoder_G.pkl'
        decoder_D_path = model_save_dir / 'decoder_D.pkl'
        torch.save(self._shared_encoder.state_dict(), shared_encoder_path)
        torch.save(self._decoder_G.state_dict(), decoder_G_path)
        torch.save(self._decoder_D.state_dict(), decoder_D_path)

    def restore(self, model_save_dir):
        shared_encoder_path = model_save_dir / 'shared_encoder.pkl'
        decoder_G_path = model_save_dir / 'decoder_G.pkl'
        decoder_D_path = model_save_dir / 'decoder_D.pkl'
        self._shared_encoder.load_state_dict(torch.load(shared_encoder_path))
        self._decoder_G.load_state_dict(torch.load(decoder_G_path))
        self._decoder_D.load_state_dict(torch.load(decoder_D_path))
