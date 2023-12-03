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


def func_a(cluster, config: Config, train_data_item, test_data_item):
    total_train_time = 0
    model = SDFVAE(
        s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
        hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
        n=feature_dim, enc_dec='CNN', nonlinearity=None)

    # train
    print(f'-------training for cluster {cluster["label"]}---------')
    x_train_list = []
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess(train_data_item, test_data_item)
    x_train_list.append(x_train)
    (config.save_dir/ f'cluster_{cluster["label"]}').mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    train_start = time.time()

    model.fit(x_train_list, save_path, max_epoch=config.max_epochs,cluster_id=cluster['label'])
    train_end = time.time()
    total_train_time += train_end - train_start

    # test
    print(f'-------testing for cluster {cluster["center"]}---------')     

    save_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
    model.restore(save_path)
    (config.result_dir/f'{cluster["center"]}').mkdir(parents=True, exist_ok=True)       
    score, recon_mean, recon_std = model.predict(x_test)
    if score is not None:
        np.save(config.result_dir/f'{cluster["center"]}/test_score.npy', -score)
        np.save(config.result_dir/f'{cluster["center"]}/recon_mean.npy', recon_mean)
        np.save(config.result_dir/f'{cluster["center"]}/recon_std.npy', recon_std)
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

