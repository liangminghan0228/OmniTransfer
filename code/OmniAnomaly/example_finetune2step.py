import os
import numpy as np
import torch
from omnianomaly.model_lgssm_concat import OmniAnomaly
import time, random
from data_config import *

class Config:
    # model parameters
    x_dims = feature_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    # outputs config
    exp_dir = exp_dir
    save_dir = exp_dir / 'model'
    base_dir = noshare_save_dir / 'model'
    result_dir = exp_dir / 'result'

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
        print(f'-------finetune for cluster {cluster["label"]}---------')
        cluster_baseline=get_offline_pvt_baseline(dataset_type, cluster['train'][:train_num])       
        for test_id in cluster['test']:
            torch_seed()
            if dataset_type == 'data2':
                preprocess_train_days = 7
                day_points = 96       
            else:
                preprocess_train_days = 5
                day_points = 288
            print(f'-------finetune for machine {test_id}---------')       
         
            new_model = OmniAnomaly(x_dims=config.x_dims,
                        z_dims=config.z_dims,
                        max_epochs=config.max_epochs//2,
                        batch_size=config.batch_size,
                        window_size=config.window_size,
                        learning_rate=global_learning_rate)
            
            # restore model
            base_path = config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
            new_model.restore(save_path=base_path)
          
            if if_freeze_seq:
                new_model.freeze_layers(freeze_layer='seq')# freeze
            if 'freeze' in exp_key:
                if 'rnn' in exp_key:
                    new_model.freeze_layers('rnn')
                # if 'dense' in exp_key:
                #     new_model.freeze_layers('dense')
                # if 'pnf' in exp_key:
                #     new_model.freeze_layers('pnf')
            # init random
            if 'init_dense_std' in exp_key:
                new_model.init_layers("std")            
            elif 'init_dense' in exp_key:
                new_model.init_layers("dense")
            
            train_data_item_raw, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
     
            train_data_item = train_data_item_raw
            x_train, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            
            train_start = time.time()
            (config.save_dir/ f'machine_{test_id}').mkdir(parents=True, exist_ok=True)
            save_path = config.save_dir/ f'machine_{test_id}' / 'model.pkl'
            valid_portion = 0.4 if training_period <= 2 and dataset_type is None else 0.3
      
            new_model.fit([x_train[-1*training_period*day_points:]], save_path=save_path, valid_portion=valid_portion, if_wrap=True,test_id=None)
            new_model.restore(save_path)
            new_model.unfreeze_layers()
            new_model.fit([x_train[-1*training_period*day_points:]], save_path=save_path, valid_portion=valid_portion, if_wrap=True,test_id=None)

            train_one_model_time = time.time() - train_start
            total_train_time += train_one_model_time
            
            print(f"train_one_model_time: {train_one_model_time} s")
       
            new_model.restore(save_path)

            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)       
            score, recon_mean, recon_std, z = new_model.predict(x_test, save_path)
            if score is not None:
                np.save(config.result_dir/f'{test_id}/test_score.npy', -score)
                np.save(config.result_dir/f'{test_id}/recon_mean.npy', recon_mean)
                np.save(config.result_dir/f'{test_id}/recon_std.npy', recon_std)
                np.save(config.result_dir/f'{test_id}/z.npy', z)

    
    print(f'exp {dataset_type} {exp_key} total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    main()
