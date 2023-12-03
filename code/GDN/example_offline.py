
import os
import numpy as np
import torch
from util.evaluate import get_err_scores
from models.TrainGDN import TrainGDN
import time, random

from data_config import *

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index


train_config = {
    'batch': global_batch_size,
    'epoch': global_epochs,
    'slide_win': global_window_size,
    'dim': global_embed_dim,
    'slide_stride': 1,
    'comment': '',
    'seed': seed,
    'out_layer_num': global_out_layer_num,
    'out_layer_inter_dim': global_out_layer_inter_dim,
    'decay': 0,
    'val_ratio': 0.3,
    'topk': global_topk,
}

env_config={
    'save_path': exp_dir,
    'dataset': 'msl',
    'report': 'best',
    'device': global_device,
    'load_model_path': exp_dir
}


def func_a(cluster):
    data_index = cluster['label']
    model = TrainGDN(train_config, env_config)
    model_save_dir = exp_dir / f'model/cluster_{data_index}'
    result_path = exp_dir / f"result/{data_index}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / f"model.pth"


    train_id_list = cluster['train'][:train_num]
    print(f"cluster:{cluster['label']} train_list:{train_id_list}")
    offline_data_list = get_offline_data(dataset_name=dataset_type, data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T)
        x_train = wrap_data(x_train, global_window_size)
        preprocess_offline_data_list.append(x_train)
    # preprocess_offline_data_list = np.array(preprocess_offline_data_list)
    # train
    train_start = time.time()
    model.fit(model_save_path, train_values=preprocess_offline_data_list)
    fit_one_time = time.time() - train_start
    print(f"fit cluster{data_index} cost:{fit_one_time}s")
    return fit_one_time


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
        train_time=func_a(cluster)
        print(f"{cluster['center' if 'center' in cluster else 'label']}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')


if __name__ == '__main__':
    # get config
    main()

