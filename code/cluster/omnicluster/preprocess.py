
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel, delayed
import scipy.stats as st
import time

CONFIG_PATH = 'configs'

data_preprocess_job_num = 40


def sub_remove_extreme_data(index, data, top_n, mode):
    data = np.copy(data)
    if index % 1000 == 0:
        print(f'remove_extreme_data--{index}')
    for n_index_data in data:
        if mode == 'extreme':

            half_length = int(len(n_index_data) * top_n / 2)
            sorted_args = np.argsort(n_index_data)
            n_index_data[sorted_args[:half_length]] = np.nan
            n_index_data[sorted_args[-half_length:]] = np.nan
        elif mode == 'deviation-mean':

            length = int(len(n_index_data) * top_n)
            mean = np.mean(n_index_data)
            abs_distance_mean = np.abs(n_index_data-mean)
            sorted_args = np.argsort(abs_distance_mean)
            n_index_data[sorted_args[-length:]] = np.nan
    return data



# @time_consuming('remove_extreme_data')
def remove_extreme_data(in_data, top_n, mode):
    data_list = Parallel(n_jobs=data_preprocess_job_num)(
        delayed(sub_remove_extreme_data)(index, data, top_n, mode) for index, data in enumerate(in_data))
    return deal_nan(np.array(data_list))


def sub_deal_nan(index, data):
    data = np.copy(data)
    if index % 1000 == 0:
        print(f'sub_deal_nan -- {index}')
    x = np.arange(data.shape[1])
    for n_index_data in data:
        if sum(np.isnan(n_index_data)) == n_index_data.shape:
            np.nan_to_num(n_index_data, copy=False)
        else:
            nan_index = np.where(np.isnan(n_index_data))[0]
            non_nan_index_x = np.setdiff1d(x, nan_index)
            non_nan_index_y = n_index_data[non_nan_index_x]
            n_index_data[nan_index] = np.interp(nan_index, non_nan_index_x, non_nan_index_y)
    return data


# @time_consuming('deal_nan')
def deal_nan(in_data):
    data_list = Parallel(n_jobs=data_preprocess_job_num)(
        delayed(sub_deal_nan)(index, data) for index, data in enumerate(in_data))
    return np.array(data_list)


# @time_consuming('moving_average')
def moving_average(in_data, window=12, min_periods=1):
    in_data = np.copy(in_data)
    for data in in_data:
        for i, n_index_data in enumerate(data):
            df = pd.Series(n_index_data)
            # moving_avg = df.rolling(window=window, min_periods=min_periods).mean()
            moving_avg = df.rolling(window=window, min_periods=min_periods).median()
            moving_avg = moving_avg.values.flatten()
            data[i] = moving_avg
    return in_data


# @time_consuming('feature_scaling')
def feature_scaling(in_data, mode):
    in_data = np.copy(in_data)
    sca = None
    if len(in_data.shape) != 3:
        raise ValueError("Data must be a 3-D array")
    if mode == "norm":
        sca = MinMaxScaler()
    elif mode == "stand":
        sca = StandardScaler()
    for i, data in enumerate(in_data):
        data = data.transpose()
        data = sca.fit_transform(data)
        in_data[i] = data.transpose()
    return in_data


def main(paras):
    data = np.load(paras["dataset_path"]) 
    print(data.shape)
    data = np.transpose(data, [0, 2, 1])
    print(data.shape)
    preprocess_start_time = time.time()
    if paras["if_remove_extreme"]:
        data = remove_extreme_data(data, paras["extreme_per"], paras['remove_extreme_mode'])
    print('remove_extreme_data finished')
    if paras["if_moving_average"]:
        data = moving_average(data, paras["moving_window_size"], paras["min_periods"])
    print('moving_average finished')
    if paras["if_feature_scaling"]:
        data = feature_scaling(data, paras["mode"])
    print(f"preprocess_time:{time.time() - preprocess_start_time}")
    np.save(paras["save_data_path"], data)