import os
import numpy as np
import torch
from interfusion.model import InterFusion
import time, random
from data_config import *

class Config:
    x_dims = feature_dim
    z1_dims = global_z1_dim
    z2_dims = global_z2_dim
    train_max_epochs = global_train_epochs
    pretrain_max_epochs = global_pretrain_epochs
    batch_size = global_batch_size
    window_size = global_window_size

    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config):
    total_train_time = 0
    pretrain_model = InterFusion(x_dims=config.x_dims,
                z1_dims=config.z1_dims,
                z2_dims=config.z2_dims,
                max_epochs=config.train_max_epochs,
                pre_max_epochs=config.pretrain_max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=pretrain_lr,
                output_padding_list=output_padding_list)
    # train
    train_id_list = cluster['train'][:train_num]
    print(f"cluster:{cluster['label']} train_list:{train_id_list}")
    offline_data_list = get_offline_data(dataset_name=dataset_type, data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T)
        preprocess_offline_data_list.append(x_train)
    (config.save_dir/ f'cluster_{cluster["label"]}').mkdir(parents=True, exist_ok=True)
    pre_save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'pre_model.pkl'
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    train_start = time.time()

    pretrain_model.prefit(preprocess_offline_data_list, pre_save_path, valid_portion=0.01)
    model = InterFusion(x_dims=config.x_dims,
        z1_dims=config.z1_dims,
        z2_dims=config.z2_dims,
        max_epochs=config.train_max_epochs,
        pre_max_epochs=config.pretrain_max_epochs,
        batch_size=config.batch_size,
        window_size=config.window_size,
        learning_rate=train_lr,
        output_padding_list=output_padding_list)
    model.restore(pre_save_path)
    model.fit(preprocess_offline_data_list, save_path, valid_portion=0.01)
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
        print(f"{cluster['center' if 'center' in cluster else 'label']}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

