import os
import numpy as np
import torch
from sdfvae.model import SDFVAE
import time, random
from data_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

class Config:
    max_epochs = global_epochs
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config):
    total_train_time = 0
    model = SDFVAE(
        s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
        hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
        n=feature_dim, enc_dec='CNN', nonlinearity=None)

    # train
    # train_id_list = cluster['train'][:10]
    train_id_list = cluster['train'][:train_num]
    print(f"cluster:{cluster['label']} train_list:{train_id_list}")
    offline_data_list = get_offline_data(dataset_name=dataset_type, data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T)
        preprocess_offline_data_list.append(x_train)
    
    (config.save_dir/ f'cluster_{cluster["label"]}').mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    train_start = time.time()
 
    valid_portion = 0.4 if training_period <= 2 and dataset_type is None else 0.3
    print(f"valid_portion", valid_portion)
    model.fit(preprocess_offline_data_list, save_path=save_path, valid_portion=valid_portion, max_epoch=config.max_epochs)
    # model.fit(preprocess_offline_data_list, save_path=save_path, valid_portion=valid_portion, max_epoch=config.max_epochs,cluster_id=cluster['label'])
    train_end = time.time()
    total_train_time += train_end - train_start

    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    torch_seed()

    total_train_time = 0
    for cluster in clusters:
        train_time=func_a(cluster, config)
        print(f"{cluster['center' if 'center' in cluster else 'label'] }--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

