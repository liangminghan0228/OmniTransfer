import os
import numpy as np
import torch
from  torch import nn
import time, random

from model.mtad_gat import MTAD_GAT
from model.training import Trainer
from model.prediction import Predictor

from data_config import *
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

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



    for cluster in clusters:
        print(f'-------finetune for cluster {cluster["label"]}---------')
        for test_id in cluster['test']:
            torch_seed()
            if dataset_type == 'yidong':
                preprocess_train_days = 7
                day_points = 96       
            else:
                preprocess_train_days = 5
                day_points = 288
            print(f'-------finetune for machine {test_id}---------')  


            train_data_item, test_data_item = get_data_by_index(dataset_type, test_id, preprocess_train_days)
            # x_train, x_test = preprocess_meanstd(train_data_item.T, test_data_item.T)
            x_train, x_test = preprocess(train_data_item.T, test_data_item.T)
            train_dataset = SlidingWindowDataset(x_train[-1*training_period*day_points:], args.window_size, target_dims)
            train_loader, val_loader, _ = create_data_loaders(
                train_dataset, args.batch_size, args.val_split, args.shuffle_dataset
                )
            save_dir = config.save_dir/ f'{test_id}'
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer = Trainer(
                model,
                optimizer,
                args.window_size,
                feature_dim,
                target_dims,
                args.epochs//2,
                args.batch_size,
                args.lr,
                forecast_criterion,
                recon_criterion,
                True,
                save_dir,
                save_dir,
                args.print_every,
                args.log_tensorboard,
                args_summary
            )
            trainer.load(config.base_dir/ f'cluster_{cluster["label"]}'/ 'model.pt')

            if 'freeze_rnn' in exp_key:
                print('freeze_rnn')
                trainer.freeze_layers()

            if 'init_dense' in exp_key:
                print('init_dense')
                trainer.init_layers()            


            train_start = time.time()
            trainer.fit(train_loader, val_loader)
            trainer.unfreeze_layers()
            trainer.fit(train_loader, val_loader)
            train_one_model_time = time.time() - train_start
            total_train_time += train_one_model_time
            
            print(f"train_one_model_time: {train_one_model_time} s")
            plot_losses(trainer.losses, save_path=save_dir, plot=False)
            with open(config.save_dir / f'{test_id}/loss_log.txt','w') as fw_loss:
                cur_loss = trainer.losses["val_total"][0]
                for loss in trainer.losses["val_total"]:
                    fw_loss.write(str(loss))
                    if loss<=cur_loss:
                        fw_loss.write('***')
                        cur_loss = loss
                    fw_loss.write('\n')

            trainer.load(f"{save_dir}/model.pt")
            best_model = trainer.model
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
                "save_path": save_dir,
            }
            predictor = Predictor(
                best_model,
                args.window_size,
                feature_dim,
                prediction_args,
            )

            predictor.predict_anomalies(x_train, x_test, label[test_id],save_output=True)
            # Save config
            args_path = f"{save_dir}/config.txt"
            with open(args_path, "w") as f:
                json.dump(args.__dict__, f, indent=2)


    
    print(f'exp {dataset_type} {exp_key} total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    main()
