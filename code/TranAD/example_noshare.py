import os
import numpy as np
import torch
from src.models import TrainTranAD
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


def func_a(cluster, config: Config, train_data_item, test_data_item):
    total_train_time = 0
    model = TrainTranAD(
        feats=config.x_dims,
        max_epoch=config.max_epochs,
    )
    # train
    print(f'-------training for cluster {cluster["label"]}---------')
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    save_dir = config.save_dir/ f'cluster_{cluster["label"]}'
    save_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.time()
 
    model.fit(x_train, save_dir=save_dir, valid_portion=0.3)
    train_end = time.time()
    total_train_time += train_end - train_start
    # test
    print(f'-------testing for cluster {cluster["center"]}---------')     
   
    model.restore(save_dir)
    (config.result_dir/f'{cluster["center"]}').mkdir(parents=True, exist_ok=True)       
    score_G, score_GD, recon_G, recon_D = model.predict(x_test)
    np.save(config.result_dir/f'{cluster["center"]}/test_score_g.npy', score_G)
    np.save(config.result_dir/f'{cluster["center"]}/test_score_d.npy', score_GD)
    np.save(config.result_dir/f'{cluster["center"]}/recon_G.npy', recon_G)
    np.save(config.result_dir/f'{cluster["center"]}/recon_D.npy', recon_D)
    # np.save(config.result_dir/f'{cluster["center"]}/z.npy', z)
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

