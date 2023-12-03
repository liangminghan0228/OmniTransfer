import os
import numpy as np

import torch

from omnianomaly.model_lgssm_concat import OmniAnomaly
# from omnianomaly.model_vae import OmniAnomaly

import time, random

from data_config import *

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index


class Config:
    x_dims = feature_dim
    z_dims = global_z_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size

    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config, train_data_item, test_data_item):
    total_train_time = 0
    model = OmniAnomaly(x_dims=config.x_dims,
                z_dims=config.z_dims,
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=global_learning_rate)

    if if_freeze_seq:
        model.freeze_layers(freeze_layer='seq')
    
    # train
    print(f'-------training for cluster {cluster["label"]}---------')
    x_train_list = []
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, _ = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    (config.save_dir/ f'cluster_{cluster["label"]}').mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    train_start = time.time()
    
    model.fit(x_train_list, save_path, valid_portion=0.05)
    train_end = time.time()
    total_train_time += train_end - train_start

    # test
    print(f'-------testing for cluster {cluster["center"]}---------')     

    
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    model.restore(save_path)
    
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)

    (config.result_dir/f'{cluster["center"]}').mkdir(parents=True, exist_ok=True)       
    score, recon_mean, recon_std, z = model.predict(x_test, save_path)
    if score is not None:
        np.save(config.result_dir/f'{cluster["center"]}/test_score.npy', -score)
        np.save(config.result_dir/f'{cluster["center"]}/recon_mean.npy', recon_mean)
        np.save(config.result_dir/f'{cluster["center"]}/recon_std.npy', recon_std)
        np.save(config.result_dir/f'{cluster["center"]}/z.npy', z)
    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    total_train_time = 0
    for cluster in clusters:
        torch_seed()
        train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        train_time=func_a(cluster, config, train_data_item.T, test_data_item.T)
        print(f"{cluster['center']}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

