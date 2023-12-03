import itertools
import numpy as np
from omnicluster.utils import *
import copy

# @time_consuming("feature selection: GET CYCLE DATA")
def get_cycle_data(path):
    cycle = np.load(path)
    return cycle[:, 1:]


# @time_consuming("feature selection: GET CYCLE ABOVE TH")
def get_cycle_above_thred(cycle_freq, th):
    cycle = []
    for i, j in cycle_freq:
        if i != 0 and j > th:
            cycle.append(i)
    return cycle


# @time_consuming("feature selection: GET INDEX ABOVE TH")
def get_index_equal_cycle_above_thred(index_cycle, th):
    index_freq = {} 
    index_list = []
    for i, j in index_cycle:
        if j > 0:
            if i in index_freq:
                index_freq[i] += 1
            else:
                index_freq[i] = 1
    for i, j in index_freq.items():
        if j > th:
            index_list.append(i)
    index_list.sort()
    return index_list


# @time_consuming("feature selection: GET ORIGIN DATA")
def get_origin_data(path, index_list):
    data = np.load(path)[:, index_list].squeeze()
    return data


# @time_consuming("feature selection: CAL CORRELATION")
def cal_correlation(data, index_comb):
    mean_data = np.mean(data, axis=2)
    mean_data = np.expand_dims(mean_data, axis=2)
    data = data - mean_data
    index_correlation = []
    for i_data in data:
        tem_correlation = []
        for index in index_comb:
            x = i_data[index[0]]
            y = i_data[index[1]]
            if np.all(x == 0):
                x = np.ones_like(x) * 1e-9
            if np.all(y == 0):
                y = np.ones_like(y) * 1e-9
            cor = np.sum(
                np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)))
            tem_correlation.append(cor)
        index_correlation.append(np.array(tem_correlation))
    return np.array(index_correlation)

def meet_rule1(R, F, index_list, F_index_list):
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if np.sum(R[i,tmp_F_index_list]) == 0:
            return index_list[i], True
    return -1, False

def meet_rule2(R, F, index_list, F_index_list):
    exist_0_flag = False
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if not np.all(R[i, tmp_F_index_list]):
            exist_0_flag = True
            break
    if not exist_0_flag:
        return -1, False
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if np.all(R[i, tmp_F_index_list]):
            return index_list[i], True
    return -1, False

def meet_rule3(R, F, index_list, F_index_list):
    for i in F_index_list:
        tmp_F_index_list = copy.deepcopy(F_index_list)
        tmp_F_index_list.remove(i)
        if not np.all(R[i, tmp_F_index_list]):
            return False
    return True

# @time_consuming("feature selection: GET INDEX ABOVE CORRELATION THRED")
def get_index_above_corre_thred(index_comb, data_correlation, sim_th, sim_num_th, index_list, data_num):
    cor = (data_correlation > sim_th).astype(int)
    cor = np.sum(cor, axis=0)
    R = np.zeros((len(index_list), len(index_list)))
    
    for cor_item, (j, i) in zip(cor, index_comb):

        if cor_item > sim_num_th:
            R[i][j] = cor_item
            R[j][i] = cor_item

    R = R / data_num

    F = set(index_list) 
    select_F = set() 
    delete_F = set() 
  
    while len(F) > 0:

        F_index_list = [index_list.index(f) for f in F]
        select_F_index_list = [index_list.index(s_f) for s_f in select_F]
        i, if_continue = meet_rule1(R, F, index_list, F_index_list)
        if if_continue:

            F.discard(i)
            select_F.add(i)
            continue

        i, if_continue = meet_rule2(R, F, index_list, F_index_list)
        if if_continue:
            F.discard(i)
            delete_F.add(i)
            continue
        if meet_rule3(R, F, index_list, F_index_list):
            if len(select_F) == 0:
                i = list(F)[0]
            else:
                min_correlation_i_SF_sum = 100
                for f_index in F_index_list:
                    tmp_sum = np.sum(R[f_index, select_F_index_list])
                    if  tmp_sum < min_correlation_i_SF_sum:
                        min_correlation_i_SF_sum = tmp_sum
                        i = index_list[f_index]

            select_F.add(i)
            F.discard(i)

            delete_F = delete_F | F
            F.clear()
            break
        else:

            min_correlation_i_F_sum = 100
            i_index = None
            for f_index in F_index_list:
                tmp_sum = np.sum(R[f_index, F_index_list])
                if tmp_sum < min_correlation_i_F_sum:
                    min_correlation_i_F_sum = tmp_sum
                    i = index_list[f_index]
                    i_index = f_index

            S_i = set()
            for f_index in F_index_list:

                if index_list[f_index] != i and R[i_index, f_index] > 0:

                    S_i.add(index_list[f_index])

            max_correlation_j_SF_sum = 0
            S_i_index_list = [index_list.index(s) for s in S_i]
            j = None
            for s_index in S_i_index_list:
                tmp_sum = np.sum(R[s_index, select_F_index_list])
                if tmp_sum > max_correlation_j_SF_sum:
                    max_correlation_j_SF_sum = tmp_sum
                    j = index_list[s_index]
            if j is None:
                j = min(S_i)

            select_F.add(i)
            delete_F.add(j)
            F.discard(i)
            F.discard(j)


                
    print(f"F:{F}\nSF:{select_F}\nDF:{delete_F}")        

    print('row', end='\t')
    for index in index_list:
        print(index, end='\t')
    print()
    for row_index, row in enumerate(R):
        print(index_list[row_index], end='\t')
        for col_index, cor in enumerate(row):
            if cor > sim_num_th / 5711:
                print(f"\033[0;31m{round(cor, 3)}\033[0m", end='\t')
            else:
                print(round(cor, 3), end='\t')
        print()
    
    return list(select_F) 

def main(params, index_list=None):
    fs_start_time = time()
    if index_list is None:
        index_cycle = get_cycle_data(params["cycle_path"])
        index_list = get_index_equal_cycle_above_thred(index_cycle, params["index_th"])

    data = get_origin_data(params["data_path"], index_list)
    index_comb = itertools.combinations(np.arange(data.shape[1]), 2)
    index_comb = np.array([i for i in index_comb])
    data_correlation = cal_correlation(data, index_comb)
    index_to_save = get_index_above_corre_thred(index_comb, data_correlation, params["sim_th"],
                                                params["sim_num_th"], index_list, data.shape[0])
    print(f"fs time:{time() - fs_start_time}")                                                

    index_to_save.sort()

    np.save(params['index_to_save'], np.array(index_to_save))
    data_to_save = np.load(params["data_path"])[:, index_to_save].squeeze()
    np.save(params["data_save_path"], data_to_save)
