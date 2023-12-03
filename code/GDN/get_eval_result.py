from models.evaluate import bf_search
import numpy as np
import pandas as pd
from data_config import *
from tqdm import tqdm

def get_data(data_idx):
    test_score_name = "test_score"
    if dataset_type == 'data2':
        test_score = np.load(exp_dir/f'result/{data_idx}/{test_score_name}.npy')[-(96*7-global_window_size+1):]
        y_test = label[data_idx, -len(test_score):]
    
    elif dataset_type == 'data1':
        test_score = np.load(exp_dir/f'result/{data_idx}/{test_score_name}.npy')[-(2*288-global_window_size+1):]
        y_test = label[data_idx, -len(test_score):]
    
    elif dataset_type == 'ctf':
        test_score = np.load(exp_dir/f'result/{data_idx}/{test_score_name}.npy')[-(8*2880-global_window_size+1):] # ctf数据集的label长度为5*2880 - 1，并不影响砍掉第一个时间窗口
        y_test = label[data_idx, -len(test_score):]

    return test_score, None, y_test

def get_bf_by_machine_threhold(calc_latency):
    res_prefix = 'bf' if calc_latency else 'pf'
    # best_df = pd.DataFrame(columns=['cluster', 'data_idx', 'best_f1', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'threshold'])

    machine_best_df = pd.DataFrame()
    if dataset_type == 'data2':
        # tp 1 fp 2 fn 3 tn 0
        tp_fp_res = np.zeros((200, 96*7 - (global_window_size - 1)))
    elif dataset_type == 'data1':
        tp_fp_res = np.zeros((316, 2*288 - (global_window_size - 1)))
    elif dataset_type == 'ctf':
        tp_fp_res = np.zeros((533, 2880*8 - (global_window_size - 1)))

    data_list = eval_data_list
    for machine_id in tqdm(data_list):
        res_point_list = {
            'machine_id': [],
            'tp': [],
            'fp': [],
            'fn': [],
            'p': [],
            'r': [],
            'f1': [],
            'threshold': []
        }

        test_score, _, y_test = get_data(machine_id)
        test_score = np.sum(test_score, axis=-1)
        test_score = (test_score - np.min(test_score))/(np.max(test_score)-np.min(test_score))*1000

        # bf_search_max = np.percentile(test_score, 95) 
        # bf_search_min = np.percentile(test_score, 5) 
        t, th, predict = bf_search(test_score, y_test,
                            start=bf_search_min,
                            end=bf_search_max,
                            step_num=int((bf_search_max-bf_search_min)//bf_search_step_size),
                            display_freq=1000,
                            calc_latency=calc_latency)
        label_item = y_test
        predict_item = predict.astype(int)
  
        tp_index = np.where((label_item == 1) & (predict_item == 1))
        fp_index = np.where((label_item == 0) & (predict_item == 1))
        fn_index = np.where((label_item == 1) & (predict_item == 0))
        tp_fp_res[machine_id, tp_index] = 1
        tp_fp_res[machine_id, fp_index] = 2
        tp_fp_res[machine_id, fn_index] = 3
        
        res_point_list['machine_id'].append(machine_id)  
        res_point_list['tp'].append(np.sum(((label_item == 1) & (predict_item == 1)).astype(int))) 
        res_point_list['fp'].append(np.sum(((label_item == 0) & (predict_item == 1)).astype(int)))
        res_point_list['fn'].append(np.sum(((label_item == 1) & (predict_item == 0)).astype(int)))
        p = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fp'][-1]+1e-9), 4)
        r = round(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fn'][-1]+1e-9), 4)
        f1 = round(2*p*r / (p+r+1e-9), 4)
        res_point_list['p'].append(p)
        res_point_list['r'].append(r)
        res_point_list['f1'].append(f1)
        res_point_list['threshold'].append(th)  
        machine_best_df = machine_best_df.append(pd.DataFrame(res_point_list))

    (exp_dir/'evaluation_result').mkdir(exist_ok=True, parents=True)
    machine_best_df = machine_best_df.sort_values(by=['machine_id'])
    machine_best_df.to_csv(exp_dir/f'evaluation_result/{res_prefix}_machine_best_f1.csv', index=False)
    np.save(exp_dir/f'evaluation_result/{res_prefix}_tp_fp_res.npy', tp_fp_res)

def read_bf_center_result(calc_latency):
    if dataset_type is None:
        with open(project_path.parent / 'exp_data/config/cluster_data2.json') as f:
            clusters = json.load(f)
    elif dataset_type == 'data1':
        with open(project_path.parent / 'exp_data/config/cluster_data1.json') as f:
            clusters = json.load(f)
    elif dataset_type == 'ctf':
        with open(project_path.parent / 'exp_data/config/cluster_ctf.json') as f:
            clusters = json.load(f)

    center_list = []
    for c in clusters:
        center_list.append(c['center'])
    center_index = []

    print(center_list)

    res_prefix = 'bf' if calc_latency else 'pf'
    best_df = pd.read_csv(exp_dir/f'evaluation_result/{res_prefix}_machine_best_f1.csv')
    for index, row in best_df.iterrows():
        if row['machine_id'] in center_list:
            center_index.append(index)

    tp = np.sum(best_df['tp'].values[center_index])
    fp = np.sum(best_df['fp'].values[center_index])
    fn = np.sum(best_df['fn'].values[center_index])
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    print(f"center result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}")

def read_bf_all_result(calc_latency):
    test_index = []
    res_prefix = 'bf' if calc_latency else 'pf'
    best_df = pd.read_csv(exp_dir/f'evaluation_result/{res_prefix}_machine_best_f1.csv')
    tp = np.sum(best_df['tp'].values)
    fp = np.sum(best_df['fp'].values)
    fn = np.sum(best_df['fn'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    res_file = open(exp_dir/f'evaluation_result/{res_prefix}_all_res.txt', mode='a')
    print(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}")
    print(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}", file=res_file)



if __name__ == '__main__':
    print(exp_key)
    # for single_score_th in range(10, 210, 10):
    # print(f"{single_score_th}---------")
    # pf
    # get_bf_by_machine_threhold(False)
    # read_bf_all_result(False)

    # bf
    get_bf_by_machine_threhold(True)
    read_bf_all_result(True)

    # center f1
    # read_bf_center_result(True)

    # get_pot()
    # get_best_f1score_for_each_cluster()
    # get_pot_result_for_each_machine()