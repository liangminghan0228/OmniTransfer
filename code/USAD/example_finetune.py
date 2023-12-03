import os
import numpy as np
import torch
from data_config import *
# from usad.model_cnn import USAD
from usad.model_fcn import USAD
# from usad.model_fcn_copy import USAD
import time, random
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index
class Config:
    x_dims = feature_dim
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    valid_step_freq = global_valid_step_freq
    exp_dir = exp_dir
    save_dir = exp_dir/'model'
    restore_dir = noshare_save_dir/'model'
    result_dir = exp_dir/'result'

def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def freeze(model: USAD):
    if "freeze_12" in exp_key:
        model.freeze_layers("freeze_12")
    elif "freeze_1" in exp_key:
        model.freeze_layers("freeze_1")
    elif "freeze_encoder" in exp_key:
        model.freeze_layers("freeze_encoder")
    else:
        print(f"freeze error") 

def main():

    torch_seed()
    total_train_time = 0
    for cluster in clusters:
        print(f'-------testing for cluster {cluster["label"]}---------')       
        if dataset_type == 'data2':
            preprocess_train_days = 7       
        else:
            preprocess_train_days = 5
        for test_id in cluster['test']:
            print(f'-------testing for machine {test_id}---------')       
        
            new_model = USAD(x_dims=config.x_dims, batch_size=config.batch_size, z_dims=config.z_dims, window_size=config.window_size)
            # restore model
            model_restore_dir = config.restore_dir / f'cluster_{cluster["label"]}'
            new_model.restore(model_restore_dir)
            # freeze(new_model)
            
            if "freeze_12" in exp_key:
                print(f"freeze 12") 
                new_model.freeze_layers("freeze_12")
            elif "freeze_1" in exp_key:
                print(f"freeze 1") 
                new_model.freeze_layers("freeze_1")
            elif "freeze_encoder" in exp_key:
                print(f"freeze encoder") 
                new_model.freeze_layers("freeze_encoder")
            else:
                print(f"freeze error") 

            if "init_1" in exp_key:
                print(f"init_1") 
                new_model.init_param("init_1")
            elif "init_23" in exp_key:
                print(f"init_23") 
                new_model.init_param("init_23")
            elif "init_3" in exp_key:
                print(f"init_3") 
                new_model.init_param("init_3")

            model_save_dir = config.save_dir / f'machine_{test_id}'
            model_save_dir.mkdir(parents=True, exist_ok=True)
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            x_train, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            train_start = time.time()
            new_model.fit([x_train[-day_points*training_period:]], model_save_dir=model_save_dir, local_epoch=global_epochs, valid_portion=0.3)
            train_one_model_time = time.time() - train_start
            print(f'--------{train_one_model_time}----')
            total_train_time += train_one_model_time
            new_model.restore(model_save_dir)
            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)       
            test_score_d, test_score_g, recon_G, recon_G_D = np.array(new_model.predict(x_test))
            np.save(config.result_dir/f'{test_id}/test_score_d.npy', test_score_d)
            np.save(config.result_dir/f'{test_id}/test_score_g.npy', test_score_g)
            np.save(config.result_dir/f'{test_id}/recon_G.npy', recon_G)
            np.save(config.result_dir/f'{test_id}/recon_G_D.npy', recon_G_D)
            # np.save(config.result_dir/f'{test_id}/x1.npy', x1)
            # np.save(config.result_dir/f'{test_id}/x2.npy', x2)

    
    print(f'total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    main()
