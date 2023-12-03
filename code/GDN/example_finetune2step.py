
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
    'epoch': global_epochs//2,
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


def func_a(cluster,):
    if dataset_type == 'data2':
        preprocess_train_days = 7       
    else:
        preprocess_train_days = 5 
    total_train_time = 0
    for test_id in cluster['test']:
        print(f"test for cluster{cluster['label']} machine{test_id}")
        train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
        train_data_item, test_data_item = preprocess_meanstd(train_data_item.T, test_data_item.T)
        train_data_item = train_data_item[-(day_points*training_period):]
        train_data_item = wrap_data(train_data_item, global_window_size)
        model = TrainGDN(train_config, env_config)
        model_restore_dir = noshare_save_dir / f'model/cluster_{cluster["label"]}'
        model_save_dir = exp_dir / f'model/machine_{test_id}'
        model_save_dir.mkdir(parents=True, exist_ok=True)
        result_path = exp_dir / f"result/{test_id}"
        result_path.mkdir(parents=True, exist_ok=True)
        model_restore_path = model_restore_dir / f"model.pth"
        model_save_path = model_save_dir / f"model.pth"

        
        # train
        model.restore(model_restore_path)
        train_start = time.time()

        # freeze
        if 'freeze_alpha' in exp_key:
            model.freeze_layers('freeze_alpha')
        if 'freeze_embedding' in exp_key:
            model.freeze_layers('freeze_embedding')
        if "init" in exp_key:
            model.init_layers()
        # step1 finetune
        model.fit(model_save_path, train_values=train_data_item)
        fit_one_time = time.time() - train_start
        print(f"fit {test_id} cost:{fit_one_time}s")
        # step2 finetune
        model.unfreeze_layers()
        model.restore(model_save_path)
        model.fit(model_save_path, train_values=train_data_item)
        fit_one_time += time.time() - train_start
        # test
        model.restore(model_save_path)
        avg_loss, test_predicted_list, test_ground_list = model.predict(test_values=test_data_item)
        score = get_err_scores(test_predicted_list, test_ground_list)
        print(score.shape, test_predicted_list.shape, test_ground_list.shape)
        np.save(result_path/f"recon_mean.npy", test_predicted_list)
        np.save(result_path/f"test_score.npy", score)
        
        total_train_time += fit_one_time
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
        total_train_time += func_a(cluster)
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')


if __name__ == '__main__':
    # get config
    main()

