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
    base_dir = noshare_save_dir/'model'
    save_dir = exp_dir/'model'
    result_dir = exp_dir/'result'

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
        model = SDFVAE(
            s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
            hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
            n=feature_dim, enc_dec='CNN', nonlinearity=None)
        # restore model
        base_path = config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
        model.restore(save_path=base_path)
        # test
        for test_id in cluster['test']:
            print(f'-------testing for cluster {cluster["label"]} test:{test_id}---------')       
            if dataset_type == 'data2':
                preprocess_train_days = 7
            else:
                preprocess_train_days = 5
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            _, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)
            save_path =  config.result_dir/f'{test_id}/1.txt'
            score, recon_mean, recon_std = model.predict(x_test)
            if score is not None:
                np.save(config.result_dir/f'{test_id}/test_score.npy', -score)
                np.save(config.result_dir/f'{test_id}/recon_mean.npy', recon_mean)
                np.save(config.result_dir/f'{test_id}/recon_std.npy', recon_std)
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':

    # get config
    config = Config()
    main()
