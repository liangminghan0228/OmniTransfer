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
    base_dir = noshare_save_dir/'model'
    

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
        model = InterFusion(x_dims=config.x_dims,
                z1_dims=config.z1_dims,
                z2_dims=config.z2_dims,
                max_epochs=config.train_max_epochs,
                pre_max_epochs=config.pretrain_max_epochs,
                batch_size=config.batch_size,
                window_size=config.window_size,
                learning_rate=train_lr,
                output_padding_list=output_padding_list)

        # restore model
        base_path = config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
        model.restore(save_path=base_path)

        if dataset_type == 'data2':
            preprocess_train_days = 7       
        else:
            preprocess_train_days = 5       

        # test
        for test_id in cluster['test']:
            print(f'-------testing for cluster {cluster["label"]} test:{test_id}---------')       
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            _, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)
            save_path =  config.result_dir/f'{test_id}/1.txt'
            score, recon_mean, recon_std, z = model.predict(x_test, save_path)
            if score is not None:
                np.save(config.result_dir/f'{test_id}/test_score.npy', -score)
                np.save(config.result_dir/f'{test_id}/recon_mean.npy', recon_mean)
                np.save(config.result_dir/f'{test_id}/recon_std.npy', recon_std)
                np.save(config.result_dir/f'{test_id}/z.npy', z)
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':

    # get config
    config = Config()
    main()
