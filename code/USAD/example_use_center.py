import numpy as np
import torch
from data_config import *
# from usad.model_cnn2 import USAD
# from usad.model_cnn import USAD
from usad.model_fcn import USAD
import random

class Config:
    x_dims =feature_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    exp_dir = exp_dir
    save_dir = noshare_save_dir/'model'
    restore_dir = None
    result_dir = exp_dir/'result'

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
        model = USAD(x_dims=config.x_dims, batch_size=config.batch_size, z_dims=config.z_dims, window_size=config.window_size)
        model_save_dir = config.save_dir / f'cluster_{cluster["label"]}'
        model.restore(model_save_dir)
        if dataset_type == 'data2':
            preprocess_train_days = 7       
        else:
            preprocess_train_days = 5
        # test
        print(f'-------testing for cluster {cluster["label"]} test:{cluster["test"]}---------')       
        for test_id in cluster['test']:
            torch_seed()
            print(f'-------testing for machine {test_id}---------')       
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            _, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)       
            test_score_d, test_score_g, recon_G, recon_G_D = np.array(model.predict(x_test))
            np.save(config.result_dir/f'{test_id}/test_score_d.npy', test_score_d)
            np.save(config.result_dir/f'{test_id}/test_score_g.npy', test_score_g)
            np.save(config.result_dir/f'{test_id}/recon_G.npy', recon_G)
            np.save(config.result_dir/f'{test_id}/recon_G_D.npy', recon_G_D)
    
    print(f'total train time: {total_train_time}')



if __name__ == '__main__':

    # get config
    config = Config()
    main()
