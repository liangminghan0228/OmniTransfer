import os
import numpy as np
import torch
from  torch import nn
import time, random

from model.mtad_gat import MTAD_GAT
from model.training import Trainer
from model.prediction import Predictor
from model.data import SlidingWindowDataset as DataSlidingWindowDataset

from data_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

class Config:
    max_epochs = global_epochs
    exp_dir = exp_dir
    save_dir = exp_dir/'result'
    # save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config):
    total_train_time = 0

    # train
    train_id_list = cluster['train'][:train_num]
    print(f"cluster:{cluster['label']} train_list:{train_id_list}")
    offline_data_list = get_offline_data(dataset_name=dataset_type, data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T)
        preprocess_offline_data_list.append(x_train)
    
    n_features = x_train.shape[1]
    out_dim = n_features
    model = MTAD_GAT(
        n_features,
        args.window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    target_dims=None
    log_dir = config.save_dir/ f'cluster_{cluster["label"]}'
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model,
        optimizer,
        args.window_size,
        n_features,
        target_dims,
        args.epochs,
        args.batch_size,
        args.lr,
        forecast_criterion,
        recon_criterion,
        True,
        log_dir,
        log_dir,
        args.print_every,
        args.log_tensorboard,
        args_summary
    )
    print('preprocess_offline_data_list len:',len(preprocess_offline_data_list))
    train_dataset = DataSlidingWindowDataset(preprocess_offline_data_list, args.window_size+1)
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset, args.batch_size, args.val_split, args.shuffle_dataset
    )
    # train_loader = SlidingWindowDataLoader(
    #             train_dataset,
    #             batch_size=args.batch_size,
    #             shuffle=True,
    #             drop_last=False,
    #         )
    train_start = time.time()
    trainer.fit(train_loader, val_loader)
    train_end = time.time()
    total_train_time += train_end - train_start
    
    plot_losses(trainer.losses, save_path=log_dir, plot=False)
    with open(config.save_dir / f'cluster_{cluster["label"]}/loss_log.txt','w') as fw_loss:
        cur_loss = trainer.losses["val_total"][0]
        for loss in trainer.losses["val_total"]:
            fw_loss.write(str(loss))
            if loss<=cur_loss:
                fw_loss.write('***')
                cur_loss = loss
            fw_loss.write('\n')
    

    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    '''
        每个类训练一个模型
    '''
    total_train_time = 0
    for cluster in clusters:
        torch_seed()
        train_time=func_a(cluster, config)
        print(f"{cluster['center' if 'center' in cluster else 'label']}--{train_time}s")
        total_train_time+=train_time
    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

