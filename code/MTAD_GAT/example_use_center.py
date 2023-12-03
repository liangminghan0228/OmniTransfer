import os
import numpy as np
import torch
from torch import nn
from model.prediction import Predictor
from model.training import Trainer
from model.mtad_gat import MTAD_GAT
import random
from data_config import *

class Config:
    max_epochs = global_epochs
    exp_dir = exp_dir
    save_dir = exp_dir/'result'
    # save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'
    base_dir = noshare_save_dir/'result'

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
    model = MTAD_GAT(
        feature_dim,
        args.window_size,
        feature_dim,
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


    # Some suggestions for POT args
    level_q_dict = {
        "data2": (0.9999, 0.001),
        "yidong": (0.9999, 0.001),
    }
    key = args.dataset_type
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"data2":0,"yidong":0}
    reg_level = reg_level_dict[key]
    target_dims=None
    train_data_item, test_data_item = get_data_by_index(dataset_type, clusters[0]['test'][0], training_period)
    # y_test = label[cluster['center']]
    x_train, x_test = preprocess(train_data_item.T, test_data_item.T)
    train_dataset = SlidingWindowDataset(x_train, args.window_size, target_dims)
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset, args.batch_size, args.val_split, args.shuffle_dataset
        )
    trainer = Trainer(
        model,
        optimizer,
        args.window_size,
        feature_dim,
        target_dims,
        args.epochs,
        args.batch_size,
        args.lr,
        forecast_criterion,
        recon_criterion,
        True,
        Config.exp_dir/'temp',
        Config.exp_dir/'temp',
        args.print_every,
        args.log_tensorboard,
        args_summary
    )
    for cluster in clusters:

        for test_id in cluster['test']:
            torch_seed()
            save_path = config.result_dir/f'{test_id}'
            save_path.mkdir(parents=True, exist_ok=True)
            prediction_args = {
                'dataset': dataset_type,
                "target_dims": target_dims,
                'scale_scores': args.scale_scores,
                "level": level,
                "q": q,
                'dynamic_pot': args.dynamic_pot,
                "use_mov_av": args.use_mov_av,
                "gamma": args.gamma,
                "reg_level": reg_level,
                "save_path": save_path,
            }

            trainer.load(config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pt')
            best_model = trainer.model

            predictor = Predictor(
                best_model,
                args.window_size,
                feature_dim,
                prediction_args,
            )
            # predictor.load(config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pt')
            
            print(f'-------testing for cluster {cluster["label"]} test:{test_id}---------')
            if dataset_type == 'yidong':
                preprocess_train_days = 7       
            else:
                preprocess_train_days = 5

            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)

            x_train, x_test = preprocess(train_data_item.T, test_data_item.T)
            print(x_test.shape)
            predictor.predict_anomalies(x_train, x_test, label[test_id],save_output=True)


    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':

    # get config
    config = Config()
    main()
