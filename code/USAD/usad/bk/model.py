import torch
import numpy as np
from torch import nn
from torch import optim

from typing import Sequence
from usad.data import SlidingWindowDataset, SlidingWindowDataLoader


class Encoder(nn.Module):

    def __init__(self, input_dims: int, z_dims: int, nn_size: Sequence[int] = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)

        layers = []
        last_size = input_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            layers.append(nn.ReLU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, z_dims))
        layers.append(nn.ReLU())
        self._net = nn.Sequential(*layers)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        z = self._net(w)
        return z


class Decoder(nn.Module):

    def __init__(self, z_dims: int, input_dims: int, nn_size: Sequence[int] = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)
        layers = []
        last_size = z_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            layers.append(nn.ReLU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, input_dims))
        layers.append(nn.Sigmoid())
        self._net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self._net(z)
        return w


class AutoEncoder(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, w):
        code = self._encoder(w)
        reconstructed = self._decoder(code)
        return reconstructed


class USAD:

    def __init__(self, x_dims: int,
                 max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size: Sequence[int] = None,
                 decoder_nn_size: Sequence[int] = None,
                 z_dims: int = 38, window_size: int = 10,
                 valid_step_frep: int = 100):
        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = 100
        self._step = 0

        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)

        self._AE_G = AutoEncoder(self._shared_encoder, self._decoder_G)
        self._AE_D = AutoEncoder(self._shared_encoder, self._decoder_D)

        self._optimizer_G = optim.Adam(self._AE_G.parameters())
        self._optimizer_D = optim.Adam(self._AE_D.parameters())

        self.mse_left = {'AE_G': {'train': [0], 'valid': [0]},
                         'AE_D': {'train': [0], 'valid': [0]}}
        self.mse_right = {'AE_G': {'train': [0], 'valid': [0]},
                          'AE_D': {'train': [0], 'valid': [0]}}
        self.loss = {'AE_G': {'train': [0], 'valid': [0]},
                     'AE_D': {'train': [0], 'valid': [0]}}

    def fit(self, values, valid_portion=0.2):
        n = int(len(values) * valid_portion)
        train_values, valid_values = values[:-n], values[-n:]

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

        mse = nn.MSELoss()

        for epoch in range(1, self._max_epochs+1):
            for i, w in enumerate(train_sliding_window):
                w = w.view(-1, self._input_dims)

                # -------------
                # Training AE_G
                # -------------
                self._optimizer_G.zero_grad()
                w_G = self._AE_G(w)
                w_G_D = self._AE_D(w_G)
                mse_left = mse(w_G, w)
                mse_right = mse(w_G_D, w)
                loss_G = (1 / epoch) * mse_left + (1 - 1 / epoch) * mse_right

                # record loss
                self.mse_left['AE_G']['train'][-1] += mse_left.item()
                self.mse_right['AE_G']['train'][-1] += mse_right.item()
                self.loss['AE_G']['train'][-1] += loss_G.item()

                # backward and update
                loss_G.backward()
                self._optimizer_G.step()

                # -------------
                # Training AE_D
                # -------------
                self._optimizer_D.zero_grad()
                w_D = self._AE_D(w)
                w_G_D = self._AE_D(w_G.detach())
                mse_left = mse(w_D, w)
                mse_right = mse(w_G_D, w)
                loss_D = (1 / epoch) * mse_left - (1 - 1 / epoch) * mse_right

                # record loss
                self.mse_left['AE_D']['train'][-1] += mse_left.item()
                self.mse_right['AE_D']['train'][-1] += mse_right.item()
                self.loss['AE_D']['train'][-1] += loss_D.item()

                # backward and update
                loss_D.backward()
                self._optimizer_D.step()

                # valid and log
                if self._step != 0 and self._step % self._valid_step_freq == 0:
                    for w in valid_sliding_window:
                        w = w.view(-1, self._input_dims)
                        w_G = self._AE_G(w)
                        w_D = self._AE_D(w)
                        w_G_D = self._AE_D(w_G)
                        mse_G_left = mse(w_G, w)
                        mse_D_left = mse(w_D, w)
                        mse_G_right = mse(w_G_D, w)
                        mse_D_right = mse_G_right
                        loss_G = (1 / epoch) * mse_G_left - (1 - 1 / epoch) * mse_G_right
                        loss_D = (1 / epoch) * mse_D_left - (1 - 1 / epoch) * mse_D_right
                        self.mse_left['AE_G']['valid'][-1] += mse_G_left.item()
                        self.mse_right['AE_G']['valid'][-1] += mse_G_right.item()
                        self.loss['AE_G']['valid'][-1] += loss_G.item()
                        self.mse_left['AE_D']['valid'][-1] += mse_D_left.item()
                        self.mse_right['AE_D']['valid'][-1] += mse_D_right.item()
                        self.loss['AE_D']['valid'][-1] += loss_D.item()

                    print("[Epoch %d/%d][step %d]" % (epoch, self._max_epochs, self._step))
                    print(
                        "[Train] [AE_G left mse: %f right mse: %f loss: %f][[AE_D left mse: %f right mse: %f loss: %f]"
                        % (self.mse_left['AE_G']['train'][-1]/100, self.mse_right['AE_G']['train'][-1]/100,
                           self.loss['AE_G']['train'][-1] / 100, self.mse_left['AE_D']['train'][-1]/100,
                           self.mse_right['AE_D']['train'][-1]/100, self.loss['AE_D']['train'][-1] / 100,)
                    )

                    print(
                        "[Valid] [AE_G left mse: %f right mse: %f loss: %f][[AE_D left mse: %f right mse: %f loss: %f]"
                        % (self.mse_left['AE_G']['valid'][-1]/100, self.mse_right['AE_G']['valid'][-1]/100,
                           self.loss['AE_G']['valid'][-1] / 100, self.mse_left['AE_D']['valid'][-1]/100,
                           self.mse_right['AE_D']['valid'][-1]/100, self.loss['AE_D']['valid'][-1] / 100,)
                    )
                    print()

                    self.mse_left['AE_G']['train'].append(0)
                    self.mse_right['AE_G']['train'].append(0)
                    self.loss['AE_G']['train'].append(0)
                    self.mse_left['AE_D']['train'].append(0)
                    self.mse_right['AE_D']['train'].append(0)
                    self.loss['AE_D']['train'].append(0)

                    self.mse_left['AE_G']['valid'].append(0)
                    self.mse_right['AE_G']['valid'].append(0)
                    self.loss['AE_G']['valid'].append(0)
                    self.mse_left['AE_D']['valid'].append(0)
                    self.mse_right['AE_D']['valid'].append(0)
                    self.loss['AE_D']['valid'].append(0)

                self._step += 1

    def predict(self, values, alpha=0.5, beta=0.5):
        collect_scores = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size),
            batch_size=self._batch_size,
        )
        mse = nn.MSELoss(reduction='none')
        for w in test_sliding_window:
            w = w.view(-1, self._input_dims)
            w_G = self._AE_G(w)
            w_G_D = self._AE_D(w_G)
            batch_scores = alpha * mse(w_G, w) + beta * mse(w_G_D, w)
            batch_scores = batch_scores.view(-1, self._window_size, self._x_dims)
            batch_scores = batch_scores.data.numpy()
            batch_scores = np.sum(batch_scores, axis=2)
            if not collect_scores:
                collect_scores.extend(batch_scores[0])
                collect_scores.extend(batch_scores[1:, -1])
            else:
                collect_scores.extend(batch_scores[:, -1])
        return collect_scores

    def save(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        torch.save(self._shared_encoder.state_dict(), shared_encoder_path)
        torch.save(self._decoder_G.state_dict(), decoder_G_path)
        torch.save(self._decoder_D.state_dict(), decoder_D_path)

    def restore(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        self._shared_encoder.load_state_dict(torch.load(shared_encoder_path))
        self._decoder_G.load_state_dict(torch.load(decoder_G_path))
        self._decoder_D.load_state_dict(torch.load(decoder_D_path))
