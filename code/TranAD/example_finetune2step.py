import os
import numpy as np
import torch
from src.models import TrainTranAD
import time, random
from data_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

class Config:
    # model parameters
    x_dims =feature_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
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

    total_train_time = 0
    for cluster in clusters:
        if dataset_type == 'data2':
            preprocess_train_days = 7       
        else:
            preprocess_train_days = 5  
        for test_id in cluster['test']:
            torch_seed()
            model = TrainTranAD(
                feats=config.x_dims,
                max_epoch=config.max_epochs//2,
            )
            # restore model
            base_path = config.base_dir/ f'cluster_{cluster["label"]}'
            model.restore(save_dir=base_path)
            if 'freeze_encoder' in exp_key:
                model.freeze_layers('freeze_encoder')
            elif 'freeze_att' in exp_key:
                model.freeze_layers('freeze_att')

            if "init" in exp_key:
                model.init_layers()

            save_path = config.save_dir/f"machine_{test_id}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f'-------testing for cluster {cluster["label"]} test:{test_id}---------')       
            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            x_train, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            # step1
            model.fit(x_train[-(day_points*training_period):], save_path, valid_portion=0.3)
            model.unfreeze_layers()
            model.restore(save_path)
            # step2
            model.fit(x_train[-(day_points*training_period):], save_path, valid_portion=0.3)

            model.restore(save_dir=save_path)
            (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)
            score_G, score_GD, recon_G, recon_D = model.predict(x_test)
            np.save(config.result_dir/f'{test_id}/test_score_g.npy', score_G)
            np.save(config.result_dir/f'{test_id}/test_score_d.npy', score_GD)
            np.save(config.result_dir/f'{test_id}/recon_G.npy', recon_G)
            np.save(config.result_dir/f'{test_id}/recon_D.npy', recon_D)
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':

    # get config
    config = Config()
    time1 = time.time()
    main()
    print(time.time()-time1)
