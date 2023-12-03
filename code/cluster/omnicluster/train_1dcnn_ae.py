
import random
import numpy as np
import torch.nn as nn
import torch
import pathlib
# from config import *

class AE_1dcnn(nn.Module):
    def __init__(self, layer_num, channel_list, kernel_size_list, stride_list, dropout_r, output_padding_list, activation=nn.ReLU(), in_channel=None) -> None:
        super(AE_1dcnn, self).__init__()
        self.layer_num = layer_num
        self.channel_list = channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.dropout_r = dropout_r
        self.activation = activation
        self.output_padding_list = output_padding_list
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        encoder_channel_list = [in_channel]+self.channel_list
        for index in range(layer_num):
            self.encoder.add_module(
                name=f"encoder_{index}",
                module=nn.Conv1d(in_channels=encoder_channel_list[index], out_channels=encoder_channel_list[index+1],
                kernel_size=self.kernel_size_list[index],
                stride=self.stride_list[index],
                padding=0,
                )
            )
            if index < layer_num-1:
                self.encoder.add_module(name=f"dropout_{index}", module=nn.Dropout(p=self.dropout_r))
                self.encoder.add_module(name=f"act_{index}", module=activation)
        
        decoder_channel_list = self.channel_list[::-1]+[in_channel]
        for index in range(layer_num):
            self.decoder.add_module(
                name=f"decoder_{index}",
                module=nn.ConvTranspose1d(in_channels=decoder_channel_list[index], out_channels=decoder_channel_list[index+1],
                kernel_size=self.kernel_size_list[layer_num-1-index],
                stride=self.stride_list[layer_num-1-index],
                padding=0,
                output_padding=self.output_padding_list[layer_num-1-index],
                )
            )
            if index < layer_num-1:
                self.decoder.add_module(name=f"dropout_{index}", module=nn.Dropout(p=self.dropout_r))
                self.decoder.add_module(name=f"act_{index}", module=activation)
    
    def forward(self, input):
        z = self.encoder(input)
        recon = self.decoder(z)
        return z, recon



class Train_AE_1dcnn:
    def __init__(self, layer_num, channel_list, kernel_size_list, stride_list, dropout_r, output_padding_list, activation, in_channel, max_epoch, batch_size, device, learning_rate) -> None:
        super(Train_AE_1dcnn, self).__init__()
        self.model = AE_1dcnn(
            layer_num, channel_list, kernel_size_list, stride_list, dropout_r, output_padding_list, activation, in_channel
        ).to(device=device)
        self.max_epoch = max_epoch
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.schedule = None
        self.device = device
        self.loss_func = nn.MSELoss()
        self.best_loss = 99999999
    
    def fit(self, train_values, save_path):
        train_data_loader = torch.utils.data.DataLoader(dataset=train_values, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self. max_epoch+1):
            loss_epoch_sum = 0
            for data in train_data_loader:
                data = data.to(device=self.device)
                z, recon = self.model(data)
                loss = self.loss_func(data, recon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch_sum += loss.item()
            if loss_epoch_sum/len(train_data_loader) < self.best_loss:
                self.best_loss=loss_epoch_sum/len(train_data_loader)
                self.save(save_path)
                print(f"***", end="")
            print(f"{epoch}/{self.max_epoch} {loss_epoch_sum/len(train_data_loader)} batch_num:{len(train_data_loader)}")
        self.save(save_path)
    
    def predict(self, test_values):
        test_data_loader = torch.utils.data.DataLoader(dataset=test_values, batch_size=self.batch_size, shuffle=False)
        z_res, recon_res = None, None
        with torch.no_grad():
            for data in test_data_loader:
                data = data.to(device=self.device)
                z, recon = self.model(data)
                if z_res is None:
                    z_res  = z.cpu().numpy()
                    recon_res = recon.cpu().numpy()
                else:
                    z_res  = np.concatenate([z_res, z.cpu().numpy()], axis=0)
                    recon_res  = np.concatenate([recon_res, recon.cpu().numpy()], axis=0)
        return z_res, recon_res


    def save(self, save_path: pathlib.Path):
        torch.save(self.model.state_dict(), save_path/f"model.pth")

    def restore(self, save_path: pathlib.Path):
        self.model.load_state_dict(torch.load(save_path/f"model.pth"))

def torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(param: dict):
    torch_seed(2021)
    data = np.load(param["data_path"]).astype(np.float32)
    print(data.shape)
    device = torch.device(f'cuda:{0}')
    model = Train_AE_1dcnn(layer_num=param['layer_num'], 
        channel_list=param['channel_list'], 
        kernel_size_list=param['kernel_size_list'], 
        stride_list=param['stride_list'], 
        dropout_r=param['dropout_r'], 
        output_padding_list=param['output_padding_list'], 
        activation=nn.ReLU(), 
        in_channel=data.shape[1], 
        max_epoch=param['epoch'], 
        batch_size=param['batch_size'], 
        device=device, 
        learning_rate=param['lr'])

    print(model.model)
    save_path=param["save_path"]
    model.fit(data, save_path=save_path)
    model.restore(save_path)
    z, recon = model.predict(data)
    print(z.shape, recon.shape)
    np.save(save_path/'z.npy', z)
    np.save(save_path/'recon.npy', recon)

    
if __name__ == '__main__':
    main()