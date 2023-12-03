import random
import torch.nn as nn
from models.GDN import GDN
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_fc_graph_struc
from util.time import *
from models.TimeDataset import TimeDataset
from models.GDN import GDN
import time
from data_config import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
class TrainGDN:
    def __init__(self, train_config, env_config):
        self.train_config = train_config
        self.env_config = env_config
        dataset = self.env_config['dataset'] 
        self.feature_map = list(range(feature_dim))
        fc_struc = get_fc_graph_struc(dataset, self.feature_map)
        set_device(env_config['device'])
        self.device = get_device()
        self.fc_edge_index = build_loc_net(fc_struc, self.feature_map, self.feature_map)
        self.fc_edge_index = torch.tensor(self.fc_edge_index, dtype = torch.long)
        self.cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }
        edge_index_sets = []
        edge_index_sets.append(self.fc_edge_index)
        # print(f"edge_index_sets:{edge_index_sets[0].shape}")
        self.model = GDN(edge_index_sets, len(self.feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)

        self.use_index_weight = True
        if self.use_index_weight:
            self.loss_func = nn.MSELoss(reduction='none')
        else:
            self.loss_func = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=global_learning_rate, weight_decay=self.train_config['decay'])

        self.debug = [1, 1]

    def get_train_data(self, train_values):
        if type(train_values) == list:
            dataset_list = []
            for train_value in train_values:
                train_dataset_indata = construct_data(train_value, self.feature_map)
                # print(f"train_dataset_indata:{len(train_dataset_indata)} {train_dataset_indata[0].shape}")
                dataset_list.append(TimeDataset(train_dataset_indata, self.fc_edge_index, mode='train', config=self.cfg))
            self.train_dataset = ConcatDataset(dataset_list)
        else:
            train_dataset_indata = construct_data(train_values, self.feature_map)
            # print(f"train_dataset_indata:{len(train_dataset_indata)} {len(train_dataset_indata[0])}")
            self.train_dataset = TimeDataset(train_dataset_indata, self.fc_edge_index, mode='train', config=self.cfg)
        self.train_dataloader, self.val_dataloader = self.get_loaders(self.train_dataset, self.train_config['batch'], val_ratio = self.train_config['val_ratio'])

    def get_test_data(self, test_values):
        test_dataset_indata = construct_data(test_values, self.feature_map)
        self.test_dataset = TimeDataset(test_dataset_indata, self.fc_edge_index, mode='test', config=self.cfg)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.train_config['batch'], shuffle=False, num_workers=0)
        return self.test_dataloader

    def valid_log(self, save_path, i_epoch, epochs, acu_loss, log_file, dataloader, min_loss):
        val_loss, _, _ = self.predict(dataloader=self.val_dataloader)
        if val_loss < min_loss:
            self.save(save_path)
            min_loss = val_loss
            print(f"***", end="")
            print(f"***", end="", file=log_file)
        print(f'epoch ({i_epoch} / {epochs}) (Loss:{acu_loss/len(dataloader)}, loss_sum:{acu_loss}), val_loss:{val_loss}')
        print(f'epoch ({i_epoch} / {epochs}) (Loss:{acu_loss/len(dataloader)}, loss_sum:{acu_loss}), val_loss:{val_loss}', file=log_file)

    def fit(self, save_path: pathlib.Path, train_values):
        print(f"fit train_values:{np.array(train_values).shape}")
        index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        self.get_train_data(train_values)
        train_loss_list = []
        acu_loss = 0
        min_loss = 1e+8
        epochs = self.train_config['epoch']
        self.model.train()
        dataloader = self.train_dataloader
        log_path = save_path.parent / f"log.txt"
        # if log_path.exists():
        #     os.remove(log_path)
        log_file = open(log_path, mode='a')
        self.valid_log(save_path, 0, epochs, acu_loss, log_file, dataloader, min_loss)
        for i_epoch in range(1, epochs+1):
            acu_loss = 0
            self.model.train()
            for x, labels, edge_index in dataloader:
                x, labels, edge_index = [item.float().to(global_device) for item in [x, labels, edge_index]]
                self.optimizer.zero_grad()
                out = self.model(x, edge_index).float().to(global_device)
                loss = self.loss_func(out, labels)

                if self.use_index_weight:
                    loss = torch.mean(index_loss_weight_tensor * loss)

                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
                acu_loss += loss.item()
            if i_epoch % global_valid_epoch_freq == 0:
                self.valid_log(save_path, i_epoch, epochs, acu_loss, log_file, dataloader, min_loss)
        return train_loss_list
    
    def predict(self, test_values=None, dataloader=None):
        index_loss_weight_tensor = torch.tensor(index_loss_weight).to(device=global_device)
        if test_values is not None:
            dataloader = self.get_test_data(test_values)
        elif dataloader is not None:
            pass
        else:
            print(f"test error")
        # test
        device = self.device
        test_loss_list = []
        now = time.time()
        test_predicted_list = []
        test_ground_list = []
        t_test_predicted_list = []
        t_test_ground_list = []
        test_len = len(dataloader)
        self.model.eval()
        i = 0
        acu_loss = 0
        for x, y, edge_index in dataloader:
            x, y, edge_index = [item.to(device).float() for item in [x, y, edge_index]]
            with torch.no_grad():
                predicted = self.model(x, edge_index).float().to(device)
                loss = self.loss_func(predicted, y)
                if self.use_index_weight:
                    loss = torch.mean(index_loss_weight_tensor * loss)
                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                else:
                    t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
            test_loss_list.append(loss.item())
            acu_loss += loss.item()
            i += 1
            if i % 10000 == 1 and i > 1:
                print(timeSincePlus(now, i / test_len))
        test_predicted_list = t_test_predicted_list.cpu().numpy()
        test_ground_list = t_test_ground_list.cpu().numpy()
        avg_loss = sum(test_loss_list)/len(test_loss_list)
        return avg_loss, test_predicted_list, test_ground_list

    # def get_loaders(self, train_dataset, batch, val_ratio=0.1):
    #     dataset_len = int(len(train_dataset))
    #     train_use_len = int(dataset_len * (1 - val_ratio))
    #     val_use_len = int(dataset_len * val_ratio)
    #     val_start_index = random.randrange(train_use_len)
    #     indices = torch.arange(dataset_len)

    #     train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    #     train_subset = Subset(train_dataset, train_sub_indices)

    #     val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    #     val_subset = Subset(train_dataset, val_sub_indices)


    #     train_dataloader = DataLoader(train_subset, batch_size=batch,
    #                             shuffle=True)

    #     val_dataloader = DataLoader(val_subset, batch_size=batch,
    #                             shuffle=False)

    #     return train_dataloader, val_dataloader

    def get_loaders(self, train_dataset, batch, val_ratio=0.1):
        all_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        dataset_len = len(all_dataloader)
        if dataset_len > 1:
            val_index_list = range(0, dataset_len, int(1//val_ratio))
            train_data = [d for i, d in enumerate(all_dataloader) if i not in val_index_list]
            val_data = [d for i, d in enumerate(all_dataloader) if i in val_index_list]
        else:
            train_data = all_dataloader
            val_data = all_dataloader
        return train_data, val_data

    
    def freeze_layers(self, freeze_layer: str = None):
        if freeze_layer == 'freeze_alpha':
            for t_gnn in self.model.gnn_layers:
                t_gnn.gnn.att_i.requires_grad = False
                t_gnn.gnn.att_j.requires_grad = False
                t_gnn.gnn.att_em_i.requires_grad = False
                t_gnn.gnn.att_em_j.requires_grad = False
        if freeze_layer == 'freeze_embedding':
            freeze_a_layer(self.model.embedding)

    def unfreeze_layers(self):
        for t_gnn in self.model.gnn_layers:
            t_gnn.gnn.att_i.requires_grad = True
            t_gnn.gnn.att_j.requires_grad = True
            t_gnn.gnn.att_em_i.requires_grad = True
            t_gnn.gnn.att_em_j.requires_grad = True
        unfreeze_a_layer(self.model.embedding)

    def init_layers(self):
        # init 全连接层的参数
        self.model.out_layer.mlp[0].reset_parameters()
        for t_gnn in self.model.gnn_layers:
            t_gnn.gnn.lin.reset_parameters()

    def save(self, save_path: pathlib.Path):
        torch.save(self.model.state_dict(), save_path)

    def restore(self, save_path: pathlib.Path):
        self.model.load_state_dict(torch.load(save_path))