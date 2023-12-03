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


def func_a(cluster, config, train_data_item, test_data_item):
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
    print(f'-------training for cluster {cluster["label"]}---------')
    x_train_list = []
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    (config.save_dir/ f'cluster_{cluster["label"]}').mkdir(parents=True, exist_ok=True)
    pre_save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'pre_model.pkl'
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    train_start = time.time()

    pretrain_model.prefit(x_train_list, pre_save_path, valid_portion=0.01)
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
    model.fit(x_train_list, save_path, valid_portion=0.01)
    train_end = time.time()
    total_train_time += train_end - train_start

    # test
    print(f'-------testing for cluster {cluster["center"]}---------')     

    
    # save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    model.restore(save_path)
    
    # x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)

    (config.result_dir/f'{cluster["center"]}').mkdir(parents=True, exist_ok=True)       
    # score, recon_mean, recon_std, z = pretrain_model.predict(x_test, save_path, if_pretrain=True)
    score, recon_mean, recon_std, z = model.predict(x_test, save_path, if_pretrain=False)
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
    torch_seed()

    total_train_time = 0

    # ts_list = Parallel(n_jobs=1)(delayed(func_a)(cluster, config, train_data[cluster['center']].T, test_data[cluster['center']].T) for cluster in clusters)
    # total_train_time += sum(ts_list)
    for cluster in clusters:
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

