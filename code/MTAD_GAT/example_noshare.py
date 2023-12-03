import os
import numpy as np
import torch
from  torch import nn
import time, random

from model.mtad_gat import MTAD_GAT
from model.training import Trainer
from model.prediction import Predictor

from data_config import *
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_index

class Config:
    max_epochs = global_epochs
    exp_dir = exp_dir
    save_dir = exp_dir/'result'
    # save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config: Config, train_data_item, test_data_item,y_test):
    total_train_time = 0
    n_features = train_data_item.shape[1]
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

    # train
    print(f'-------training for cluster {cluster["label"]}---------')
    x_train_list = []
    train_id = cluster["center"]
    print(f'---------machine index: {train_id}---------')
    x_train, x_test = preprocess(train_data_item, test_data_item)
    x_train_list.append(x_train)
    log_dir = config.save_dir/ f'{cluster["label"]}'
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir/ f'{cluster["label"]}'/ 'model.pkl'

    train_dataset = SlidingWindowDataset(x_train, args.window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, args.window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, args.batch_size, args.val_split, args.shuffle_dataset, test_dataset=test_dataset
    )
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
    

    train_start = time.time()
    trainer.fit(train_loader, val_loader)
    train_end = time.time()
    total_train_time += train_end - train_start

    plot_losses(trainer.losses, save_path=log_dir, plot=False)
    with open(config.save_dir / f'{cluster["label"]}/loss_log.txt','w') as fw_loss:
        cur_loss = trainer.losses["val_total"][0]
        for loss in trainer.losses["val_total"]:
            fw_loss.write(str(loss))
            if loss<=cur_loss:
                fw_loss.write('***')
                cur_loss = loss
            fw_loss.write('\n')


    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

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

    trainer.load(f"{log_dir}/model.pt")
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
        "save_path": log_dir,
    }
    best_model = trainer.model
    print(args.window_size,
                n_features,
                prediction_args,)
    predictor = Predictor(
        best_model,
        args.window_size,
        n_features,
        prediction_args,
    )

    predictor.predict_anomalies(x_train, x_test, y_test,save_output=True)
    # Save config
    args_path = f"{log_dir}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    torch_seed()
    '''
        每个类训练一个模型
    '''
    total_train_time = 0

    for cluster in clusters:

        train_data_item, test_data_item = get_data_by_index(dataset_type, cluster['center'], training_period)
        y_test = label[cluster['center']]
        print(train_data_item.shape,test_data_item.shape)
        train_time=func_a(cluster, config, train_data_item.T, test_data_item.T,y_test)
        print(f"{cluster['center']}--{train_time}s")
        total_train_time+=train_time


    print(f'exp:{dataset_type}{exp_key}total train time: {total_train_time}')



if __name__ == '__main__':
    # get config
    config = Config()
    print(config.save_dir)
    main()

