import torch
import numpy as np
from torch import nn
from torch import optim
from typing import List, Sequence
from usad.data import SlidingWindowDataset, SlidingWindowDataLoader
from data_config import *
import copy

# def init_layer(layer: nn.Module):
#     nn.init.constant_(layer.bias, 0)

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False

def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
act = nn.LeakyReLU(0.2)

class Encoder(nn.Module):

    def __init__(self, input_dims: int, z_dims: int, nn_size: Sequence[int] = None, freeze_index_list: List =[]):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dims, input_dims // 2),
            copy.deepcopy(act),
            nn.Linear(input_dims // 2, input_dims // 4),
            copy.deepcopy(act),
            nn.Linear(input_dims // 4, z_dims),
        )
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        z = self._net(w)
        # print(f"encoder z:{z.shape}")
        return z


class Decoder(nn.Module):
    def __init__(self, z_dims: int, input_dims: int, nn_size: Sequence[int] = None, freeze_index_list: List =[]):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(z_dims, input_dims // 4),
            copy.deepcopy(act),
            nn.Linear(input_dims // 4, input_dims // 2),
            copy.deepcopy(act),
            nn.Linear(input_dims // 2, input_dims),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self._net(z)
        # print(f"decoder w:{w.shape}")
        return w


class USAD:

    def __init__(self, x_dims, batch_size, z_dims, window_size):
        self._x_dims = x_dims
        self._batch_size = batch_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = global_valid_step_freq
        self._step = 0
     
        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims).to(global_device)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims).to(global_device)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims).to(global_device)
        self.loss_func = nn.MSELoss(reduction='none')
        self._optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, list(self._shared_encoder.parameters()) + list(self._decoder_G.parameters()) ), lr=global_lr)
        self._optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, list(self._shared_encoder.parameters()) + list(self._decoder_D.parameters()) ), lr=global_lr)
        self.loss = {'AE_G': {'train': [0], 'valid': [0]}, 'AE_D': {'train': [0], 'valid': [0]}}
        self.valid_value = [0]
        self.min_wgd_w_delta = 99999999
        
    def freeze_layers(self, name):
        if name == "freeze_1":
            freeze_a_layer(self._shared_encoder._net[0])
            freeze_a_layer(self._decoder_G._net[4])
            freeze_a_layer(self._decoder_D._net[4])
        elif name == 'freeze_12':
            freeze_a_layer(self._shared_encoder._net[0])
            freeze_a_layer(self._shared_encoder._net[2])
            freeze_a_layer(self._decoder_G._net[4])
            freeze_a_layer(self._decoder_G._net[2])
            freeze_a_layer(self._decoder_D._net[4])
            freeze_a_layer(self._decoder_D._net[2])
        elif name == 'freeze_encoder':
            freeze_a_layer(self._shared_encoder._net[0])
            freeze_a_layer(self._shared_encoder._net[2])
            freeze_a_layer(self._shared_encoder._net[4])
        else:
            print(f"name:{name} error")

    def unfreeze_layers(self, name):
        if name == "freeze_1":
            unfreeze_a_layer(self._shared_encoder._net[0])
            unfreeze_a_layer(self._decoder_G._net[4])
            unfreeze_a_layer(self._decoder_D._net[4])
        elif name == 'freeze_12':
            unfreeze_a_layer(self._shared_encoder._net[0])
            unfreeze_a_layer(self._shared_encoder._net[2])
            unfreeze_a_layer(self._decoder_G._net[4])
            unfreeze_a_layer(self._decoder_G._net[2])
            unfreeze_a_layer(self._decoder_D._net[4])
            unfreeze_a_layer(self._decoder_D._net[2])
        elif name == 'freeze_encoder':
            unfreeze_a_layer(self._shared_encoder._net[0])
            unfreeze_a_layer(self._shared_encoder._net[2])
            unfreeze_a_layer(self._shared_encoder._net[4])
        else:
            print(f"name:{name} error")

    def init_param(self,name):
        # if name == 'init_3':
        #     init_layer(self._shared_encoder._net[4])
        #     init_layer(self._decoder_G._net[0])
        #     init_layer(self._decoder_D._net[0])
        if name == 'init_1':
            self._shared_encoder._net[0].reset_parameters()
            self._decoder_G._net[4].reset_parameters()
            self._decoder_D._net[4].reset_parameters()
        elif name == 'init_3_decoder':
            self._decoder_G._net[0].reset_parameters()
            self._decoder_D._net[0].reset_parameters()    
        elif name == 'init_3':
            self._shared_encoder._net[4].reset_parameters()
            self._decoder_G._net[0].reset_parameters()
            self._decoder_D._net[0].reset_parameters()
        elif name == 'init_23':
            self._shared_encoder._net[2].reset_parameters()
            self._shared_encoder._net[4].reset_parameters()
            self._decoder_G._net[0].reset_parameters()
            self._decoder_G._net[2].reset_parameters()
            self._decoder_D._net[0].reset_parameters()
            self._decoder_D._net[2].reset_parameters()

    def compose_loss(self, w, w_G, w_G_D, w_D, epoch, loss_type='train'):
        index_loss_weight_tensor = torch.tensor(index_loss_weight).to(global_device)
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
    def valid_log(self, valid_data, model_save_dir, log_file, epoch):
        for w in valid_data:
            w = w.view(-1, self._input_dims)
            w = w.to(global_device)
            w = w.unsqueeze(dim=1)
            z = self._shared_encoder(w)
            w_G = self._decoder_G(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G))
            self.valid_value[-1] += torch.sum(self.loss_func(w, w_G_D)).cpu().detach().item()
        if self.valid_value[-1] < self.min_wgd_w_delta:
            self.min_wgd_w_delta = self.valid_value[-1]
            self.save(model_save_dir)
            print(f"***", end=' ')
            print(f"***", end=' ', file=log_file)
        print(f"epoch:{epoch} step:{self._step} train loss_G:{self.loss['AE_G']['train'][-1]} loss_D:{self.loss['AE_D']['train'][-1]} valid w-wgd:{self.valid_value[-1]}")
        print(f"epoch:{epoch} step:{self._step} train loss_G:{self.loss['AE_G']['train'][-1]} loss_D:{self.loss['AE_D']['train'][-1]} valid w-wgd:{self.valid_value[-1]}", file=log_file)


    def fit(self, values, model_save_dir: pathlib.Path, local_epoch, valid_portion=0.3):
        train_values = values
        if_drop_last = False if (dataset_type == 'data2' and training_period <= 2) else True
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size, if_wrap=True),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=if_drop_last
        )
        all_data = list(train_sliding_window)
        if len(all_data)>1:
            valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
            train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
            train_data = np.array(all_data)[train_index_list]
            valid_data = np.array(all_data)[valid_index_list]
        else:
           
            train_data = all_data
            valid_data = all_data

        log_path = model_save_dir / f"log.txt"
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')
        
        self.valid_log(valid_data, model_save_dir, log_file, 0)
        for epoch in range(1, local_epoch+1):
            for i, w in enumerate(train_data):
                w = w.to(global_device)
                w = w.view(-1, self._input_dims)
                w = w.unsqueeze(dim=1)
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
            if epoch % global_valid_epoch_freq == 0:
                self.valid_log(valid_data, model_save_dir, log_file, epoch)
                self.loss['AE_G']['train'].append(0)
                self.loss['AE_D']['train'].append(0)
                self.loss['AE_G']['valid'].append(0)
                self.loss['AE_D']['valid'].append(0)
                self.valid_value.append(0)

                # self._step += 1
                # if self._step % learning_rate_decay_by_step == 0:
                #     for p in self._optimizer_G.param_groups:
                #         p['lr'] *= learning_rate_decay_factor
                #     for p in self._optimizer_D.param_groups:
                #         p['lr'] *= learning_rate_decay_factor
        self.valid_log(valid_data, model_save_dir, log_file, epoch)

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
                w = w.to(global_device)
                w = w.view(-1, self._input_dims)
                w = w.unsqueeze(dim=1)

                z = self._shared_encoder(w)
                w_G = self._decoder_G(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G))
                batch_scores_g = self.loss_func(w_G, w)
                batch_scores_d = self.loss_func(w_G_D, w)
                batch_scores_d = batch_scores_d.view(-1, self._window_size, self._x_dims)
                batch_scores_g = batch_scores_g.view(-1, self._window_size, self._x_dims)
                batch_scores_d = batch_scores_d.cpu().numpy()
                batch_scores_g = batch_scores_g.cpu().numpy()
                w_G = w_G.view(-1, self._window_size, self._x_dims).cpu().numpy()
                w_G_D = w_G_D.view(-1, self._window_size, self._x_dims).cpu().numpy()
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
