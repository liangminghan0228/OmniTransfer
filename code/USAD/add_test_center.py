import os
import pickle
import logging
import numpy as np

import torch

from usad.model_fcn import USAD
from usad.evaluate import bf_search
from usad.utils import get_data, ConfigHandler, merge_data_to_csv
from sklearn.preprocessing import MinMaxScaler


from data_config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from joblib import delayed, Parallel

class Config:
    # dataset configuration
    dataset = 'new'

    # model parameters
    x_dims = 19
    max_train_size = None  # `None` means full train set
    max_test_size = None
    train_start = 0
    test_start = 0
    max_epochs = 250
    batch_size = global_batch_size
    window_size = global_window_size
    z_dims = global_z_dim
    encoder_nn_size = None  # `None` means that nn_size is `(input_dims // 2, input_dims // 4)`
    deocder_nn_size = None
    valid_step_freq = 200
    alpha = global_alpha
    beta = global_beta

    # outputs config
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    restore_dir = None
    result_dir = exp_dir/'result'
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'

    # evaluation parameters
   
    bf_search_min = 0.9
    bf_search_max = 1.04
    bf_search_step_size = 0.00001

def main():
    for cluster in clusters:
        model = USAD(x_dims=config.x_dims,
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                z_dims=config.z_dims,
                window_size=config.window_size,
                valid_step_frep=config.valid_step_freq,
                freeze_index_list_encoder=freeze_index_list_encoder,
                freeze_index_list_decoder=freeze_index_list_decoder
                )

        shared_encoder_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'shared_encoder.pkl'
        decoder_G_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'decoder_G.pkl'
        decoder_D_path = config.save_dir/ f'cluster_{cluster["label"]}'/ 'decoder_D.pkl'
        model.restore(shared_encoder_path, decoder_G_path, decoder_D_path)

        test_id = cluster['center']
        x_train, x_test = preprocess_meanstd(train_data[test_id].T, test_data[test_id].T)
        test_score = np.array(model.predict(x_test, alpha=config.alpha, beta=config.beta, on_dim=True))
        (config.result_dir/f'{test_id}').mkdir(parents=True, exist_ok=True)       
        np.save(config.result_dir/f'{test_id}'/f'test_score.npy', test_score)
        recon_G, recon_G_D, z = np.array(model.reconstruct(x_test))
        np.save(config.result_dir/f'{test_id}/recon_G.npy', recon_G)
        np.save(config.result_dir/f'{test_id}/recon_G_D.npy', recon_G_D)
        np.save(config.result_dir/f'{test_id}/z.npy', z)


if __name__ == '__main__':

    # get config
    config = Config()
    main()
