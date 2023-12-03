from sdfvae.evaluate import bf_search
import numpy as np
import pandas as pd
from data_config import *
from tqdm import tqdm

prefix=''

def get_data(data_idx):
    # train_score = np.load(exp_dir/f'{data_idx}/train_score.npy')
    # test_score_name = "test_score_g"
    # test_score_name = "test_score"
    test_score_name = f"{prefix}test_score"
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



def get_bf_by_cluster_threhold(calc_latency):
    res_prefix = 'bf' if calc_latency else 'pf'
    best_df = pd.DataFrame(columns=['cluster', 'data_idx', 'best_f1', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'threshold'])

    machine_best_df = pd.DataFrame()
    if dataset_type is None:
        # tp 1 fp 2 fn 3 tn 0
        tp_fp_res = np.zeros((200, 96*7 - (global_window_size - 1))) 
    elif dataset_type == 'data1':
        tp_fp_res = np.zeros((316, 2*288 - (global_window_size - 1)))
    elif dataset_type == 'ctf':
        tp_fp_res = np.zeros((523, 2880*8 - (global_window_size - 1)))

    for cluster in clusters:
        res_point_list = {
            'machine_id': [],
            'tp': [],
            'fp': [],
            'fn': [],
            'p': [],
            'r': [],
            'f1': [],
        }
        test_score_list, y_test_list = [], []
        for data_idx in cluster['test']:
            test_score, _, y_test = get_data(data_idx)
            # test_score_clip = np.where(test_score > -single_score_th, test_score, -single_score_th)
            # get the joint score
            test_score_list.append(np.sum(test_score, axis=-1))
            # get best f1
            y_test_list.append(y_test[-len(test_score):])

        t, th, predict = bf_search(np.hstack(test_score_list), np.hstack(y_test_list),
                            start=bf_search_min,
                            end=bf_search_max,
                            step_num=int(abs(bf_search_max - bf_search_min) /
                                        bf_search_step_size),
                            display_freq=1000,
                            calc_latency=calc_latency)
        best_df.loc[best_df.shape[0],:] = [str(cluster['label'])]+[str(cluster['test'])]+list(t[:-1])+[th]

        item_length = eval_item_length
        for machine_index, machine_id in enumerate(cluster['test']):
      
            # label_item = (np.hstack(y_test_list))[machine_index*item_length:(machine_index+1)*item_length]
            label_item = y_test_list[machine_index]
            predict_item = predict[machine_index*item_length:(machine_index+1)*item_length].astype(int)
          
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
            res_point_list['p'].append(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fp'][-1]+1e-9))
            res_point_list['r'].append(res_point_list['tp'][-1] / (res_point_list['tp'][-1]+res_point_list['fn'][-1]+1e-9))
            res_point_list['f1'].append(2*res_point_list['p'][-1]*res_point_list['r'][-1] / (res_point_list['p'][-1]+res_point_list['r'][-1]+1e-9))
            # print(f"tp:{res_point_list['tp'][-1]}")
        
        res_machine_path = exp_dir/ f'{prefix}evaluation_result/best_f1_result/{cluster["label"]}/{res_prefix}_best_f1_machine.csv'
        res_machine_path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(res_point_list).to_csv(res_machine_path, index=False)
        machine_best_df = machine_best_df.append(pd.DataFrame(res_point_list))
    # best_df.loc['mean', :] = best_df.mean()
    (exp_dir/f'{prefix}evaluation_result').mkdir(exist_ok=True, parents=True)
    best_df.to_csv(exp_dir/f'{prefix}evaluation_result/{res_prefix}_best_f1.csv', index=False)
    machine_best_df = machine_best_df.sort_values(by=['machine_id'])
    machine_best_df.to_csv(exp_dir/f'{prefix}evaluation_result/{res_prefix}_machine_best_f1.csv', index=False)
    np.save(exp_dir/f'{prefix}evaluation_result/{res_prefix}_tp_fp_res.npy', tp_fp_res)

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

    (exp_dir/f'{prefix}evaluation_result').mkdir(exist_ok=True, parents=True)
    machine_best_df = machine_best_df.sort_values(by=['machine_id'])
    machine_best_df.to_csv(exp_dir/f'{prefix}evaluation_result/{res_prefix}_machine_best_f1.csv', index=False)
    np.save(exp_dir/f'{prefix}evaluation_result/{res_prefix}_tp_fp_res.npy', tp_fp_res)

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
    best_df = pd.read_csv(exp_dir/f'{prefix}evaluation_result/{res_prefix}_machine_best_f1.csv')
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
    best_df = pd.read_csv(exp_dir/f'{prefix}evaluation_result/{res_prefix}_machine_best_f1.csv')
    tp = np.sum(best_df['tp'].values)
    fp = np.sum(best_df['fp'].values)
    fn = np.sum(best_df['fn'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    res_file = open(exp_dir/f'{prefix}evaluation_result/{res_prefix}_res.txt',mode='a')
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
    # prefix = 'weighted_'
    # get_bf_by_machine_threhold(True)
    # read_bf_all_result(True)
    # center f1
    # read_bf_center_result(True)

    # get_pot()
    # get_best_f1score_for_each_cluster()
    # get_pot_result_for_each_machine()