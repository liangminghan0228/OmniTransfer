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
    base_dir = noshare_save_dir / 'model'

def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    if dataset_type == 'data2':
        preprocess_train_days = 7       
    else:
        preprocess_train_days = 5 
    torch_seed()
    total_train_time = 0
    for cluster in clusters:
        print(f'-------finetune for cluster {cluster["label"]}---------')       
        
        for test_id in cluster['test']:
            print(f'-------finetune for machine {test_id}---------')       
          
            new_model = SDFVAE(
                s_dim=global_s_dim, d_dim=global_d_dim, conv_dim=global_conv_dim, 
                hidden_dim=global_hidden_dim, T=global_T, w=global_window_size, 
                n=feature_dim, enc_dec='CNN', nonlinearity=None)
            
            # restore model
            base_path = config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pkl'
            new_model.restore(save_path=base_path)
        
            if if_freeze_seq:
                new_model.freeze_layers(name='seq')# freeze
            if 'freeze' in exp_key:
                if 'rnn' in exp_key:
                    new_model.freeze_layers('rnn')
                if 'cnn' in exp_key:
                    new_model.freeze_layers('cnn')
            # # init random
            if 'init' in exp_key:
                if 'init_dense_mean_std' in exp_key:
                    new_model.init_layers('mean_std')  
                elif 'init_dense_std' in exp_key:
                    new_model.init_layers('std')                
                elif 'init_dense' in exp_key:
                    new_model.init_layers('dense')
                elif "init_decoder_linear" in exp_key:
                    new_model.init_layers("decoder_std_linear")
            
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            x_train, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            
            train_start = time.time()
            (config.save_dir/ f'machine_{test_id}').mkdir(parents=True, exist_ok=True)
            save_path = config.save_dir/ f'machine_{test_id}' / 'model.pkl'
            valid_portion = 0.4 if training_period <= 2 and dataset_type is None else 0.3

            new_model.fit(x_train[-(day_points*training_period):], save_path=save_path, valid_portion=valid_portion, max_epoch=config.max_epochs//2)
            new_model.restore(save_path)
            new_model.lr = new_model.lr / 2.5
            from torch import optim
            new_model.optimizer = optim.Adam(new_model.vae.parameters(), new_model.lr)
            new_model.unfreeze_layers('cnn')
            new_model.unfreeze_layers('rnn')
            new_model.fit(x_train[-(day_points*training_period):], save_path=save_path, valid_portion=valid_portion, max_epoch=config.max_epochs//2)
            
            train_one_model_time = time.time() - train_start
            total_train_time += train_one_model_time
            
            print(f"train_one_model_time: {train_one_model_time} s")
            
            new_model.restore(save_path)


            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)       
            score, recon_mean, recon_std = new_model.predict(x_test)
            if score is not None:
                np.save(config.result_dir/f'{test_id}/test_score.npy', -score)
                np.save(config.result_dir/f'{test_id}/recon_mean.npy', recon_mean)
                np.save(config.result_dir/f'{test_id}/recon_std.npy', recon_std)

    
    print(f'exp {dataset_type} {exp_key} total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    main()
