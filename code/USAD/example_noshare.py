import numpy as np
import torch
from data_config import *
# from usad.model_cnn import USAD
# from usad.model_fcn import USAD
from usad.model_fcn import USAD
import time, random

class Config:
    x_dims = feature_dim
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'

def func_a(cluster, config, train_data_item, test_data_item):
    total_train_time = 0
    model = USAD(x_dims=config.x_dims, batch_size=config.batch_size, z_dims=config.z_dims, window_size=config.window_size)
    print(f'-------training for cluster {cluster["label"]}---------')
    x_train_list = []
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess_meanstd(train_data_item, test_data_item)
    x_train_list.append(x_train)
    model_save_dir = config.save_dir / f'cluster_{cluster["label"]}'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.time()
    model.fit(x_train_list, model_save_dir, local_epoch=global_epochs)
    train_end = time.time()
    total_train_time += train_end - train_start
    # test
    print(f'-------testing for cluster {cluster["center"]}---------')       
    # # restore model
    model.restore(model_save_dir)
    (config.result_dir/f'{cluster["center"]}').mkdir(parents=True, exist_ok=True)       
    test_score_d, test_score_g, recon_G, recon_G_D,x1,x2, z = np.array(model.predict(x_test))
    x1 = np.array(x1)
    x2 = np.array(x2)
    z = np.array(z)
    print(f"x1:{x1.shape} x2:{x2.shape} z:{z.shape}")
    np.save(config.result_dir/f'{cluster["center"]}/test_score_d.npy', test_score_d)
    np.save(config.result_dir/f'{cluster["center"]}/test_score_g.npy', test_score_g)
    np.save(config.result_dir/f'{cluster["center"]}/recon_G.npy', recon_G)
    np.save(config.result_dir/f'{cluster["center"]}/recon_G_D.npy', recon_G_D)
    np.save(config.result_dir/f'{cluster["center"]}/x1.npy', x1)
    np.save(config.result_dir/f'{cluster["center"]}/x2.npy', x2)
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
 
    total_train_time = 0
    for cluster in clusters:
        torch_seed()
        train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        train_time=func_a(cluster, config, train_data_item.T, test_data_item.T)
        print(f"{cluster['center']}--{train_time}s")
        total_train_time+=train_time
    print(f'total train time: {total_train_time}')



if __name__ == '__main__':
    config = Config()
    print(config.save_dir)
    main()

