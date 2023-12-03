import numpy as np
from joblib import Parallel, delayed
yin_job_num = 30
from time import time
from tqdm import tqdm


def diff_func(data, max_win):
    # equation (6)
    df = [0] * max_win
    if isinstance(data, list):
        data_length = len(data)
    elif isinstance(data, np.ndarray):
        data_length = data.shape[0]
    else:
        raise NotImplementedError("Unsupported data type")
    for win in range(1, max_win):
        for now_data in range(0, data_length - max_win):
            df[win] += np.power(data[now_data] - data[now_data + win], 2)
    return df

def get_cmndf(df):
    # equation (8)
    if isinstance(df, list):
        data_length = len(df)
    elif isinstance(df, np.ndarray):
        data_length = df.shape[0]
    else:
        raise NotImplementedError("Unsupported data type")
    cmndf = [1.0] * data_length
    for now_data in range(1, data_length):
        if np.sum(df[:now_data + 1]) != 0:
            cmndf[now_data] = df[now_data] * (now_data + 1) / np.sum(df[:now_data + 1]).astype(float)
    return cmndf

def get_pitch(cmndf, min_win, max_win, th):
    for win in range(min_win, max_win):
        if cmndf[win] < th:
            if win + 1 < max_win and cmndf[win + 1] < cmndf[win]:
                continue
            return win
    return 0

def get_yin(data, win_min, win_max, th):
    # Compute YIN
    # print(f"get_yin data:{data.shape}")
    # if np.all(np.array(data) == 0):
    #     return 0
    df = diff_func(data, win_max)
    # if np.all(np.array(df) == 0):
    #     return 0
    cmndf = get_cmndf(df)
    # print(f"cmndf:{cmndf[win_min]}")
    p = get_pitch(cmndf, win_min, win_max, th)

    return (data.shape[0] // p if p != 0 else 0), cmndf[win_min]

# @time_consuming('sub_get_all_cycle')
def sub_get_all_cycle(index_ins, data_ins, win_min, win_max, th):
    if index_ins % 300 == 0:
        print(f"sub_get_all_cycle index_ins:{index_ins}")
    cycle = []
    cmndf_list = []
    for index_index, data_index in enumerate(data_ins):
        yin, cmndf = get_yin(data_index, win_min, win_max, th)
        cycle.append([index_ins, index_index, yin])
        cmndf_list.append(cmndf)
    # print(f"cycle:{len(cycle)} cmndf_list:{len(cmndf_list)}")
    return cycle, cmndf_list

# @time_consuming("GET CYCLE")
def get_all_cycle(data, win_min, win_max, th):
    cycle = []
    cmndf_array = np.zeros((data.shape[0], data.shape[1]))
    # print(f"cmndf_array:{cmndf_array.shape}")
    cycle_cmndf_list = Parallel(n_jobs=yin_job_num)(
        delayed(sub_get_all_cycle)(index_ins, data_ins, win_min, win_max, th) for index_ins, data_ins in tqdm(enumerate(data)))

    for index, (c, cmndf) in enumerate(cycle_cmndf_list):
        cycle.extend(c)
        cmndf_array[index] = np.array(cmndf)
    return np.array(cycle), cmndf_array

def main(params):
    try:
        data = np.squeeze(np.load(params["data_path"]), axis=-1)
    except:
        data = np.load(params["data_path"])
    yin_start_time = time()
    cycle, cmndf = get_all_cycle(data, params["win_min"], params["win_max"], params["th"])
    print(f'yin time:{time() - yin_start_time}')
    np.save(params["cycle_path"], cycle)
    np.save(params["cmndf_path"], cmndf)
