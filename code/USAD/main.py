import os
import pickle
import logging
import numpy as np

import torch

from usad.model_fcn import USAD
from usad.evaluate import bf_search
from usad.utils import get_data, ConfigHandler, merge_data_to_csv

torch.cuda.set_device(1)
def main():
    logging.basicConfig(
            level='INFO',
            format='[%(asctime)s [%(levelname)s]] %(message)s')
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size,
                 train_start=config.train_start, test_start=config.test_start, do_preprocess=True)

    model = USAD(x_dims=config.x_dims, max_epochs=config.max_epochs,
                 batch_size=config.batch_size, z_dims=config.z_dims,
                 window_size=config.window_size,
                 valid_step_frep=config.valid_step_freq)

    # restore model
    if config.restore_dir:
        print(f'Restore model from `{config.restore_dir}`')
        shared_encoder_path = os.path.join(config.restore_dir, 'shared_encoder.pkl')
        decoder_G_path = os.path.join(config.restore_dir, 'decoder_G.pkl')
        decoder_D_path = os.path.join(config.restore_dir, 'decoder_D.pkl')
        model.restore(shared_encoder_path, decoder_G_path, decoder_D_path)

    # train model
    else:
        model.fit(x_train)

    # save model
    if config.save_dir:
        shared_encoder_path = os.path.join(config.save_dir, 'shared_encoder.pkl')
        decoder_G_path = os.path.join(config.save_dir, 'decoder_G.pkl')
        decoder_D_path = os.path.join(config.save_dir, 'decoder_D.pkl')
        model.save(shared_encoder_path, decoder_G_path, decoder_D_path)



    # _1, _2 = model.reconstruct(x_test)
    # with open('AE_1.pkl', 'wb') as file:
    #     pickle.dump(_1, file)
    # with open('AE_2.pkl', 'wb') as file:
    #     pickle.dump(_2, file)

    # another predict way

    # new_scores = model.predict_mean(x_test)
    # # new_scores[np.where(new_scores > 1)] = 1
    # print(new_scores.shape)
    # with open('new_score.pkl', 'wb') as file:
    #     pickle.dump(new_scores, file)
    # local = model.localization(new_scores)
    # with open('local.pkl', 'wb') as file:
    #     pickle.dump(local, file)
    # with open('mean.pkl', 'wb') as file:
    #     pickle.dump(mean, file)
    # with open('var.pkl', 'wb') as file:
    #     pickle.dump(var, file)

    train_score = model.predict(x_train, on_dim=True)
    if config.train_score_filename:
        with open(os.path.join(config.result_dir, config.train_score_filename), 'wb') as file:
            pickle.dump(train_score, file)

    test_score = model.predict(x_test, on_dim=True)
    if config.test_score_filename:
        with open(os.path.join(config.result_dir, config.test_score_filename), 'wb') as file:
            pickle.dump(test_score, file)
    # score = np.append(train_score, test_score)
    # print('size of score: ', score.shape)
    # print('min and max of score: ', np.min(score), np.max(score))
    # np.savetxt(os.path.join(config.result_dir, 'label.csv'), score,  header='tag', delimiter=',')
    # merge_data_to_csv(x_train, x_test, score, os.path.join(config.result_dir, f'{config.dataset}.csv'))

    # evaluate model by best f1-score, precision and, recall
    print('search min', config.bf_search_min)
    print('search max', config.bf_search_max)
    print('search step size', config.bf_search_step_size)
    t, th = bf_search(test_score, y_test[-len(test_score):],
                      start=config.bf_search_min,
                      end=config.bf_search_max,
                      step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                                   config.bf_search_step_size),
                      display_freq=50)



if __name__ == '__main__':

    # get config
    config = ConfigHandler().config
    print('Configuration:')
    print(config)
    main()
