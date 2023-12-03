from sdfvae.data import *
import torch
from torch import optim
import torch.nn as nn
from sdfvae.vae import VAE
from data_config import *
from torch.distributions import Normal

class SDFVAE(nn.Module):
    def __init__(self, s_dim=8, d_dim=10, conv_dim=100, hidden_dim=40,
                 T=20, w=36, n=24, enc_dec='CNN', nonlinearity=None) -> None:
        super(SDFVAE, self).__init__()
        self.vae = VAE(s_dim=s_dim, d_dim=d_dim, conv_dim=conv_dim, hidden_dim=hidden_dim,
                 T=T, w=w, n=n, enc_dec=enc_dec, nonlinearity=nonlinearity).to(global_device)
        self.window_size = w
        self.batch_size = global_batch_size
        self.T  = T
        self.loss = {'train': [], 'valid': [], 'recon': []}
        self.best_valid_loss = 99999999
        self.best_recon = 99999999
        self.lr = global_learning_rate
        self.optimizer = optim.Adam(self.vae.parameters(), self.lr)
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=40, eta_min=g_min_lr)
    
    def freeze_layers(self, name):
        if name == 'cnn':
            for param in self.vae.conv.parameters():
                param.requires_grad = False
            for param in self.vae.deconv_mu.parameters():
                param.requires_grad = False
            for param in self.vae.deconv_logsigma.parameters():
                param.requires_grad = False 
        elif name == 'rnn':
            for param in self.vae.d_lstm_prior.parameters():
                param.requires_grad = False
            for param in self.vae.s_lstm.parameters():
                param.requires_grad = False
            for param in self.vae.d_rnn.parameters():
                param.requires_grad = False
    def unfreeze_layers(self, name):
        if name == 'cnn':
            for param in self.vae.conv.parameters():
                param.requires_grad = True
            for param in self.vae.deconv_mu.parameters():
                param.requires_grad = True
            for param in self.vae.deconv_logsigma.parameters():
                param.requires_grad = True 
        elif name == 'rnn':
            for param in self.vae.d_lstm_prior.parameters():
                param.requires_grad = True
            for param in self.vae.s_lstm.parameters():
                param.requires_grad = True
            for param in self.vae.d_rnn.parameters():
                param.requires_grad = True

    def init_layers(self, init_layer: str = None):
   
        if init_layer == 'cnn':
            self.vae.conv.reset_parameters()
            self.vae.deconv_mu.reset_parameters()
            self.vae.deconv_logsigma.reset_parameters()

        elif init_layer == 'rnn':
            self.vae.d_lstm_prior.reset_parameters()
            self.vae.s_lstm.reset_parameters()
            self.vae.d_rnn.reset_parameters()
        elif init_layer == 'dense':
            print(f"init dense")
            self.vae.phi_d_prior[0].reset_parameters()
            self.vae.enc_d_prior[0].reset_parameters()
            self.vae.enc_d_prior[2].reset_parameters()
            self.vae.phi_conv[0].reset_parameters()
            self.vae.phi_conv[2].reset_parameters()
            self.vae.phi_d[0].reset_parameters()
            self.vae.enc_d[0].reset_parameters()
            self.vae.enc_d[2].reset_parameters()

            self.vae.d_mean_prior.reset_parameters()
            self.vae.d_logvar_prior.reset_parameters()
            self.vae.s_mean.model[0].reset_parameters()
            self.vae.s_logvar.model[0].reset_parameters()
            self.vae.d_mean.reset_parameters()
            self.vae.d_logvar.reset_parameters()
        elif init_layer == 'std':
            print(f"init std")

            # self.vae.d_mean_prior.reset_parameters()
            # self.vae.d_logvar_prior.reset_parameters()
            # self.vae.s_mean.model[0].reset_parameters()
            self.vae.s_logvar.model[0].reset_parameters()
            # self.vae.d_mean.reset_parameters()
            self.vae.d_logvar.reset_parameters()
        elif init_layer == 'mean_std':
            print(f"init mean_std")

            # self.vae.d_mean_prior.reset_parameters()
            # self.vae.d_logvar_prior.reset_parameters()
            self.vae.s_mean.model[0].reset_parameters()
            self.vae.s_logvar.model[0].reset_parameters()
            self.vae.d_mean.reset_parameters()
            self.vae.d_logvar.reset_parameters()
        elif init_layer == 'decoder_std_linear':
            print(f"init decoder_std_linear")
            self.vae.deconv_fc_logsigma[0].model[0].reset_parameters()
            self.vae.deconv_fc_logsigma[1].model[0].reset_parameters()
        else:
            print(f"no {init_layer}")


    def get_data(self, values, valid_portion, max_epochs):
        # if_drop_last = False if (dataset_type == 'data2' and training_period <= 2) else True
        train_values = values
        print(f"all:{train_values[0].shape}")
        # print(f"dataset:{SlidingWindowDataset(train_values, self.window_size)._strided_values.shape}")
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self.window_size, self.T, if_wrap=True),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        all_data = list(train_sliding_window)
        # print(f"all_data:{len(all_data)} {all_data[0].shape}")
        print(f"len(all_data):{len(all_data)}")
        if len(all_data) > 1:
            valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
            train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
            train_data = np.array(all_data)[train_index_list]
            valid_data = np.array(all_data)[valid_index_list]
        else: 
            train_data = all_data
            valid_data = all_data
        # self.valid_step_freq = len(train_data)*(max_epochs) // 3+1
        # self.valid_step_freq = len(train_data) * 5
        # self.valid_step_freq = len(train_data)
        # self.valid_step_freq = 30
        return train_data, valid_data
    @staticmethod
    def loss_fn(original_seq, recon_seq_mu, recon_seq_logsigma, s_mean,
                s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar,cluster_id):
        batch_size = original_seq.size(0)

        index_loss_weight_tensor = torch.tensor(index_loss_weight).unsqueeze(dim=-1).to(device=global_device)
        # print(f"original_seq.shape:{original_seq.shape} recon_seq_mu:{recon_seq_mu.shape}")
        loglikelihood = -0.5 * torch.sum(
            (torch.pow(((original_seq.float() - recon_seq_mu.float()) / torch.exp(recon_seq_logsigma.float())), 2) + 
            2 * recon_seq_logsigma.float()+ np.log(np.pi * 2)).mul(index_loss_weight_tensor)/torch.sum(index_loss_weight_tensor)
            )
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (7) for details.
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (6) for details.
        d_post_var = torch.exp(d_post_logvar)
        d_prior_var = torch.exp(d_prior_logvar)
        kld_d = 0.5 * torch.sum(d_prior_logvar - d_post_logvar
                                + ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var)
                                - 1)
        # loss, llh, kld_s, kld_d
        return (-loglikelihood + kld_s + kld_d) / batch_size, -loglikelihood / batch_size, kld_s / batch_size, kld_d / batch_size

    def valid_log(self, valid_data, log_file, save_path, epoch, max_epochs,cluster_id):
        for w in valid_data:
            w = w.to(global_device).permute(0, 1, 3, 2)
            s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logsigma = None,None,None,None,None,None,None,None,None,None
            s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logsigma = self.vae(w)                    
            valid_loss, recon, kls, kld = SDFVAE.loss_fn(w, recon_x_mu, recon_x_logsigma, s_mean, s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar,cluster_id)
            self.loss['valid'][-1] += valid_loss.item()
            self.loss['recon'][-1] += recon.item()
            del s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logsigma, recon, kls, kld
            torch.cuda.empty_cache()

        # if self.loss['valid'][-1] <= self.best_valid_loss:
        if self.loss['recon'][-1] <= self.best_recon:
            self.best_valid_loss = self.loss['valid'][-1]
            self.best_recon = self.loss['recon'][-1]
            if self.best_valid_loss < -1000:
                return True            
          
            self.save(save_path)
            print("***", end=' ')
            print("***", end=' ', file=log_file)
        if np.isnan(self.loss['valid'][-1]):
            print(f"self.loss['valid'][-1]:{self.loss['valid'][-1]}")
            return True
        
        print(f"epoch:{epoch}/{max_epochs} step:{self.step} valid_loss:{self.loss['valid'][-1]} valid_recon:{self.loss['recon'][-1]} train_loss:{self.loss['train'][-1]}")
        print(f"epoch:{epoch}/{max_epochs} step:{self.step} valid_loss:{self.loss['valid'][-1]} valid_recon:{self.loss['recon'][-1]} train_loss:{self.loss['train'][-1]}", file=log_file)
        return False


    def fit(self,  values, save_path: pathlib.Path, max_epoch, cluster_id=None,valid_portion=0.3):
        if_break = False
        self.step = 0
        train_data, valid_data = self.get_data(values=values, valid_portion=valid_portion, max_epochs=max_epoch)
        # print(f"train:{train_data[0].shape}")            
        if len(valid_data)>0:
            print(f"valid:{valid_data[0].shape}")
        # print(train_data, valid_data)
        
        log_path = save_path.parent / 'log.txt'
        # if log_path.exists():
        #     os.remove(log_path)
        log_file = open(log_path, mode='a')
        self.loss['train'].append(0)
        self.loss['valid'].append(0)
        self.loss['recon'].append(0)   
        self.valid_log(valid_data, log_file, save_path, 0, max_epoch,cluster_id)
        for epoch in range(1, max_epoch+1):
            # print("Running Epoch : {}".format(epoch + 1))
            for i, data in enumerate(train_data):
                self.step += 1
                self.loss['train'].append(0)
                self.loss['valid'].append(0)
                self.loss['recon'].append(0)
                data = data.to(global_device).permute(0, 1, 3, 2)
                self.optimizer.zero_grad()
                s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logsigma = self.vae(data)
                loss, llh, kld_s, kld_d = SDFVAE.loss_fn(data, recon_x_mu, recon_x_logsigma, s_mean, s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar,cluster_id)
                loss.backward()
                self.optimizer.step()
                self.loss['train'][-1]+=loss.item()
                if self.step % learning_rate_decay_by_step == 0:
                    for p in self.optimizer.param_groups:
                        p['lr'] *= learning_rate_decay_factor
            if epoch % global_valid_epoch == 0:
                if_break = self.valid_log(valid_data, log_file, save_path, epoch, max_epoch,cluster_id)
            if if_break:
                break
            # self.schedule.step()

        self.loss['valid'].append(0)
        self.loss['recon'].append(0)
        self.valid_log(valid_data, log_file, save_path, max_epoch, max_epoch,cluster_id)

    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logsigma):
        llh = -0.5 * (
            torch.pow(((x.float() - recon_x_mu.float()) / torch.exp(recon_x_logsigma.float())), 2)
                               + 2 * recon_x_logsigma.float()
                               + np.log(np.pi * 2))
        return llh

        # p_recon = Normal(loc=recon_x_mu, scale=torch.exp(recon_x_logsigma))
        # return p_recon.log_prob(x)

    def predict(self, values):
        self.vae.eval()
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self.window_size, self.T),
            batch_size=self.batch_size,
        )
        score = []
        recon_mean = []
        recon_std = []
        for i, data in enumerate(test_sliding_window):
            data = data.to(global_device).permute(0, 1, 3, 2)
            s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logsigma = self.vae(data)
            # print(data[:, -1, :, -1].shape, recon_x_mu[:, -1, -1, :, -1].shape, recon_x_logsigma[:, -1, -1, :, -1].shape)
            llh_last_timestamp = self.loglikelihood_last_timestamp(data[:, -1, :, -1], recon_x_mu[:, -1, :, -1], recon_x_logsigma[:, -1, :, -1])
            score.extend(llh_last_timestamp.detach().cpu().numpy())
            recon_mean.extend(recon_x_mu[:, -1, :, -1].detach().cpu().numpy())
            recon_std.extend(torch.exp(recon_x_logsigma)[:, -1, :, -1].detach().cpu().numpy())
        score = np.array(score)
        recon_mean = np.array(recon_mean)
        recon_std = np.array(recon_std)
        print(f"score:{score.shape} recon_mean:{recon_mean.shape} recon_std:{recon_std.shape}")
        return score, recon_mean, recon_std

    def save(self, save_path: pathlib.Path):
        torch.save(self.vae.state_dict(), save_path)

    def restore(self, save_path: pathlib.Path):
        self.vae.load_state_dict(torch.load(save_path))