import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        # print(f"edge_index:{edge_index.shape}")
        self.mode = mode
        data = torch.tensor(raw_data).double()
        self.x, self.y= self.process(data)
        # print(f"self.x:{self.x.shape} self.y:{self.y.shape}")
    
    def __len__(self):
        return len(self.x)

    def process(self, data):
        x_arr, y_arr = [], []
        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'
        node_num, total_time_len = data.shape
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        for i in rang:
            ft = data[:, i-slide_win:i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        return x, y

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()

        return feature, y, edge_index





