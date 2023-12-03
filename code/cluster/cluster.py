#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pathlib
project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
sys.path.append(project_path / "code/cluster")
import omnicluster.feature_selection as fs
import omnicluster.preprocess as pre
import omnicluster.yin as yin
import omnicluster.train_1dcnn_ae as ae
from joblib import Parallel, delayed
import sklearn.cluster as sc
from collections import Counter
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str)
args = parser.parse_args()

isSmooth = False
weight_coef = 1
outlier_threshold = 5
addWeight = True
dataset_name = args.data_type
if dataset_name=='data2':
    moving_window_size = 4
elif dataset_name == 'data1':
    moving_window_size = 12
# moving_window_size = 4
group_key = 'cluster'
ma_size = 5


data_root = project_path/'code/cluster'/f'out_cluster/{dataset_name}'
save_path = pathlib.Path(data_root) / group_key
save_path.mkdir(parents=True, exist_ok=True)
(save_path / 'pred_res').mkdir(parents=True, exist_ok=True)
(save_path / 'cluster_json').mkdir(parents=True, exist_ok=True)
(save_path / f'anomaly_detection_{group_key}').mkdir(parents=True, exist_ok=True)
(save_path / 'label_jsons').mkdir(parents=True, exist_ok=True)

distance_strategy = 'euc'
cluster_dir = group_key
prefix = "_pvt"
# prefix = ""

test_dataset_path = project_path / f"code/test_dataset/{dataset_name}" 
test_dataset_path.mkdir(parents=True, exist_ok=True)

online_data_path = project_path / f"dataset/{dataset_name}/online_data.npy"
offline_data_path = project_path / f'dataset/{dataset_name}/offline_data.npy'
label_path = project_path / f"dataset/{dataset_name}/label.npy"

online_data = np.load(online_data_path)
offline_data = np.load(offline_data_path)
label = np.load(label_path)
np.save(test_dataset_path/'label.npy',label)
np.save(test_dataset_path/'online_data.npy',online_data)
online_data = np.transpose(online_data,[0,2,1])

def moving_average(in_data, window=12, min_periods=1):
    for data in in_data:
        for i, n_index_data in enumerate(data):
            df = pd.Series(n_index_data)
            moving_avg = df.rolling(window=window, min_periods=min_periods).mean()
            data[i] = moving_avg
    return in_data
if isSmooth==True:
    online_data = np.transpose(online_data,[0,2,1])
    # (200, 38, 1344)
    print(online_data.shape)
    moving_average(online_data,ma_size)
    online_data = np.transpose(online_data,[0,2,1])
    offline_data = np.transpose(offline_data,[0,2,1])
    # (200, 38, 1344)
    print(offline_data.shape)
    moving_average(offline_data,ma_size)
    offline_data = np.transpose(offline_data,[0,2,1])


online_data_path = save_path / f"online_data.npy"
offline_data_path = save_path / f"offline_data.npy"
np.save(online_data_path,online_data)
np.save(offline_data_path,offline_data)

param = {
    "dataset_path": offline_data_path,
    "if_remove_extreme": True,
    "extreme_per": 0.05,
    "remove_extreme_mode": "deviation-mean",
    "if_moving_average": True,
    "moving_window_size": moving_window_size,
    "min_periods": 1,
    "if_feature_scaling": True,
    "mode": 'stand',
    "save_data_path": save_path/f"preprocess_data.npy",
}
pre.main(param)


if addWeight:
    yin_paras = {
        "data_path": save_path / 'preprocess_data.npy',
        "cycle_path": save_path / 'cycle_all.npy',
        "cmndf_path": save_path / 'cmndf.npy',
        "win_min": 96,
        "win_max": 97,
        "th": 0.3
    }
    yin.main(yin_paras)
else:
    np.save(save_path / f"cmndf.npy", np.ones((200, 25)))

cmndf = np.load(save_path / f"cmndf.npy")

np.save(save_path/f"anomaly_detection_{group_key}/cmndf.npy",cmndf)
index_to_weight_euc = 1/(cmndf**weight_coef)
index_to_weight_euc = np.mean(index_to_weight_euc, axis=0)
index_to_weight_euc = index_to_weight_euc / np.sum(index_to_weight_euc)
print(f"index_to_weight_euc:{index_to_weight_euc}")
np.save(save_path/"index_to_weight_euc.npy",index_to_weight_euc)


data = np.load(save_path /"preprocess_data.npy")
print(data.shape)
res_list = []
for i in range(200):
    for j in range(14):
        res_list.append(data[i,:,1*j*96:(1*j+1)*96])
data_arr = np.array(res_list)
print(data_arr.shape)
np.save(save_path / 'z_to_cluster_all.npy',data_arr)

index_to_weight_euc = np.load(save_path / "index_to_weight_euc.npy")
    
def euc(x, y, x_index, y_index):
    weight_item = index_to_weight_euc
    return np.sum(np.linalg.norm(x - y, axis=1)*weight_item)


def cal_distance(data):
    instance_pair_list = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            instance_pair_list.append((i, j))
    if distance_strategy == 'euc':
        # distance_list = Parallel(n_jobs=500)(delayed(euc)(data[i], data[j]) for i, j in instance_pair_list)
        distance_list = []
        from tqdm import tqdm
        for i, j in tqdm(instance_pair_list):
            distance_list.append(euc(data[i], data[j],i,j))

    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for index, dis in enumerate(distance_list):
        i, j = instance_pair_list[index]
        distance_matrix[i][j] = dis
        distance_matrix[j][i] = dis
    np.save(save_path / f'{distance_strategy}.npy', distance_matrix)

data = np.load(save_path / f'z_to_cluster_all.npy')
cal_distance(data)


index_to_weight_euc = np.load(save_path / "index_to_weight_euc.npy")
global_pvt = 0
from tqdm import tqdm
from joblib import Parallel, delayed

def ncc(x, y, x_index, y_index):
    weight_item = index_to_weight_euc
    norm_cc_list = []
    index_num, m = x.shape
    x_norm = np.linalg.norm(x, axis=1)
    for s in range(-m + 1, m):
        tmp_norm_cc_list = []
        if s >= 0:
            y_s = np.hstack((y[:, m-s:], y[:, :m - s]))
            tmp_norm_cc_list.append(
                np.sum(np.multiply(x, y_s), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)+ 1e-9)*weight_item)
        else:
            y_s = np.hstack((y[:, -s:], y[:, :-s]))
            tmp_norm_cc_list.append(
                np.sum(np.multiply(x, y_s), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)+ 1e-9)*weight_item)
        norm_cc = np.sum(tmp_norm_cc_list)
        norm_cc_list.append(norm_cc)
        ncc = np.max(norm_cc_list)
    max_index = np.argmax(norm_cc_list)
    # print(f"norm_cc_list:{norm_cc_list[max_index-1]} {norm_cc_list[max_index]} {norm_cc_list[max_index+1]}")
    s_max = list(range(-m+1, m))[max_index]
    return s_max        
    
def func_a(data_index, pvt):
    s_max = ncc(data[pvt], data[data_index], pvt, data_index)
    return s_max

def align_phase(euc_mat):
    dis_sum = np.sum(euc_mat, axis=0)
    data = np.load(save_path / 'z_to_cluster_all.npy')

    pvt = np.argmin(dis_sum)
    global_pvt = pvt
    print(pvt, dis_sum[pvt])
    # s_list = np.zeros(data.shape[0])
    s_list = Parallel(n_jobs=30)(delayed(func_a)(data_index, pvt) for data_index in tqdm(range(data.shape[0])))
    s_list = np.array(s_list)
    for data_index, s_max in enumerate(s_list):
        data[data_index] = np.concatenate([data[data_index, :, -(s_max):], data[data_index, :, 0:-(s_max)]], axis=-1)
    return s_list, data


euc_mat = np.load(save_path/'euc.npy')
s_list, data = align_phase(euc_mat)
np.save(save_path/'z_to_cluster_all_pvt.npy', data)
np.save(save_path/'s_list.npy', s_list)


def euc(x, y, x_index, y_index):
    weight_item = index_to_weight_euc
    return np.sum(np.linalg.norm(x - y, axis=1)*weight_item)
def cal_distance(data):

    instance_pair_list = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            instance_pair_list.append((i, j))
    distance_list = []

    for i, j in tqdm(instance_pair_list):
        # distance_list.append(w_distance(data[i], data[j]))
        distance_list.append(euc(data[i], data[j],i,j))

    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for index, dis in enumerate(distance_list):
        i, j = instance_pair_list[index]
        distance_matrix[i][j] = dis
        distance_matrix[j][i] = dis
    # np.save(save_path / f'w{prefix}.npy', distance_matrix)
    np.save(save_path / f'{distance_strategy}{prefix}.npy', distance_matrix)

data = np.load(save_path / f'z_to_cluster_all{prefix}.npy')
print(data.shape)
cal_distance(data)


outlier_threshold = 10
def cluster_main(distance, cluster_num):
    model = sc.AgglomerativeClustering(n_clusters=cluster_num, 
    compute_full_tree=True, 
    affinity="precomputed",
    linkage="average",
    distance_threshold=None)
    pred = model.fit_predict(distance)
    pred_list = list(Counter(pred).items())
    for pred_num in pred_list:
        if pred_num[1] <= outlier_threshold:
            pred[np.where(pred==pred_num[0])]=-1
    new_pred_list = list(Counter(pred).items())

    pred_index = [i[0] for i in new_pred_list]
    pred_index = sorted(pred_index)
    pred_index_dict = {i:j+1 for j, i in enumerate(pred_index)}
    # print(pred_index_dict)
    new_pred = np.zeros_like(pred)
    for k, v in pred_index_dict.items():
        new_pred[np.where(pred==k)] = v
    new_new_pred_list = list(Counter(new_pred).items())
    
    print(f"{len(new_new_pred_list)}--{new_new_pred_list}")
    np.save(save_path / 'pred_res' / f'cluster_pred_{cluster_num}_AgglomerativeClustering_{distance_strategy}{prefix}_{outlier_threshold}.npy', new_pred)

distance = np.load(save_path / f'{distance_strategy}{prefix}.npy')
# distance = np.load(save_path / f'all_dis_pvt.npy')
for cluster_num in [5,6,7,8,9,]:
    print(f"cluster_num:{cluster_num}")
    metric_after = cluster_main(distance, cluster_num)


cluster_num = 6
cluster_pred = np.load(save_path / f'pred_res/cluster_pred_{cluster_num}_AgglomerativeClustering_{distance_strategy}{prefix}_{outlier_threshold}.npy')
pred_list = list(Counter(cluster_pred).items())
print(pred_list)
pred_array = np.array(pred_list)
res = []
pred_label_list = pred_array[:, 0]  
subset_distance_matrix = np.load(save_path / f'{distance_strategy}{prefix}.npy')

for pred_label in pred_label_list:
    if pred_label == 1:
        continue
    item_dict = {
        'label': int(pred_label),
        'center': None,
        'train': [],
        'test': [],
        'distance': []
    }
    item_index_list = np.where(cluster_pred == pred_label)[0]

    sub_distance_matrix = subset_distance_matrix[item_index_list, :][:, item_index_list]
    sub_distance_matrix_sum = np.sum(sub_distance_matrix, axis=1)

    cluster_center_index = np.argsort(sub_distance_matrix_sum)[0]

    distance_to_center = sub_distance_matrix[cluster_center_index]
    other_node_index = np.argsort(distance_to_center)

    item_dict['center'] = int(item_index_list[cluster_center_index])

    item_dict['test'] = item_index_list[other_node_index.tolist()].tolist()

    item_dict['distance'] = subset_distance_matrix[item_dict['center'], item_dict['test']].tolist()
    
    item_dict['center'] = int(item_dict['center'])
    item_dict['train'].append(item_dict['center'])
    item_dict['test'] = item_dict['test']
    
    
    res.append(item_dict)
with open(save_path / f'cluster_json/{cluster_dir}_{cluster_num}_{distance_strategy}_{outlier_threshold}.json', mode='w') as f:
    json.dump(res, f)


online_data = np.load(online_data_path)
print(online_data.shape)

if dataset_name=='data2':
    data_1day = online_data[:,96*6:96*7,:] 
elif dataset_name == 'data1':
    data_1day = online_data[:,96*4:96*5,:] 

print(data_1day.shape)
np.save(save_path/"online_data_1day.npy",data_1day)


param = {
    "dataset_path": save_path/"online_data_1day.npy",
    "if_remove_extreme": True,
    "extreme_per": 0.05,
    "remove_extreme_mode": "deviation-mean",
    "if_moving_average": True,
    "moving_window_size": moving_window_size,
    "min_periods": 1,
    "if_feature_scaling": True,
    "mode": 'stand',
    "save_data_path": save_path/f"preprocess_onlinedata.npy",
}
pre.main(param)


z = np.load(save_path/f"preprocess_onlinedata.npy")
# fs_index = np.load(save_path/f"index_to_save.npy")
# z_to_cluster_all_online = z[:, fs_index, :]
z_to_cluster_all_online = z
print(z_to_cluster_all_online.shape)
np.save(save_path / 'z_to_cluster_all_online.npy', z_to_cluster_all_online)


index_to_weight_euc = np.load(save_path / "index_to_weight_euc.npy")

def align_phase_online():
    # data = np.load(save_path / 'z_to_cluster_all_online.npy')
    data_online = np.load(save_path / 'z_to_cluster_all_online.npy')
    data_offline = np.load(save_path / 'z_to_cluster_all_pvt.npy')

    pvt = global_pvt
    def ncc(x, y,x_index):
        weight_item = index_to_weight_euc
        norm_cc_list = []
        index_num, m = x.shape
        x_norm = np.linalg.norm(x, axis=1)
        for s in range(-m + 1, m):
            tmp_norm_cc_list = []
            if s >= 0:
                y_s = np.hstack((y[:, m-s:], y[:, :m - s]))
                tmp_norm_cc_list.append(
                    np.sum(np.multiply(x, y_s), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)+ 1e-9)*weight_item)
            else:
                y_s = np.hstack((y[:, -s:], y[:, :-s]))
                tmp_norm_cc_list.append(
                    np.sum(np.multiply(x, y_s), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1)+ 1e-9)*weight_item)
            norm_cc = np.sum(tmp_norm_cc_list)
            norm_cc_list.append(norm_cc)
        ncc = np.max(norm_cc_list)
        max_index = np.argmax(norm_cc_list)
        # print(f"norm_cc_list:{norm_cc_list[max_index-1]} {norm_cc_list[max_index]} {norm_cc_list[max_index+1]}")
        s_max = list(range(-m+1, m))[max_index]
        return s_max
    s_list = np.zeros(data_online.shape[0])
    for data_index in range(data_online.shape[0]):
        s_max = ncc(data_offline[pvt], data_online[data_index],pvt)
        s_list[data_index] = s_max
        data_online[data_index] = np.concatenate([data_online[data_index, :, -(s_max):], data_online[data_index, :, 0:-(s_max)]], axis=-1)
    return s_list, data_online
# euc_mat = np.load(save_path/'euc_online.npy')
s_list, data = align_phase_online()
print(data.shape)
np.save(save_path/'z_to_cluster_all_pvt_online.npy', data)
np.save(save_path/'s_list_online.npy', s_list)


cluster_json = json.load(open(save_path / f'cluster_json/{cluster_dir}_{cluster_num}_{distance_strategy}_{outlier_threshold}.json'))
index_to_weight_euc = np.load(save_path / "index_to_weight_euc.npy")

def euc_online(x, y, center_index,f=None):
    weight_item = index_to_weight_euc
    euc_d_item = np.linalg.norm(x - y, axis=1)*weight_item
    if f is None:
        return np.sum(euc_d_item)
    else:
        f.write(f"euc to {center_index}:{(euc_d_item/weight_item).tolist()}\n")

res = {}

for cluster in cluster_json:
    label = cluster['label']
    center = cluster['center']
    train = cluster['test']
    res[label]={"label": label, "center": center, "train": train[0:100], "test": [], "distance": []}


z_to_cluster_all_online = np.load(save_path / "z_to_cluster_all_pvt_online.npy")
z_to_cluster_all_offline = np.load(save_path/f"z_to_cluster_all_pvt.npy")
for online_index in range(z_to_cluster_all_online.shape[0]):
    dis_list = []

    for cluster in cluster_json:
        center_index = cluster['center']
        euc_d = euc_online(z_to_cluster_all_online[online_index], z_to_cluster_all_offline[center_index], center_index)
        dis_list.append(euc_d)

    min_index = 0
    min_distance = dis_list[0]
    for i,dis in enumerate(dis_list):
        if dis<min_distance:
            min_distance = dis
            min_index = i

    cluster_label = cluster_json[min_index]['label']
    res[cluster_label]['test'].append(online_index)
    res[cluster_label]['distance'].append(min_distance)
    center_index = cluster_json[min_index]['center']


res_json = []
for i,j in res.items():
    test_list = j['test']
    if len(test_list)==0:
        continue
    distance = j['distance']
    node_index = np.argsort(distance)
    sorted_test_list = sorted(test_list,key=lambda x: node_index[test_list.index(x)])
    distance.sort()
    j['test']=sorted_test_list
    res_json.append(j)
# print(res_json)
for cluster in res_json:
    print(cluster['label'],len(cluster['test']))

class NumpyEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,  
            np.int16, np.int32, np.int64, np.uint8,  
            np.uint16, np.uint32, np.uint64)):  
            return int(obj)  
        elif isinstance(obj, (np.float_, np.float16, np.float32,np.float64)):  
            return float(obj)  
        elif isinstance(obj, (np.ndarray,)):  
            return obj.tolist()  
        return json.JSONEncoder.default(self, obj) 

with open(save_path / f'cluster_json/{cluster_dir}_{cluster_num}_{distance_strategy}_online_{outlier_threshold}.json', mode='w') as f:
    json.dump(res_json, f,cls=NumpyEncoder)


with open(test_dataset_path/'cluster.json', mode='w') as f:
    json.dump(res_json, f,cls=NumpyEncoder)

data = np.load(offline_data_path)
print(data.shape)
res_list = []
for i in range(200):
    for j in range(14):
        res_list.append(data[i,j*96:(j+1)*96,:])
data_new_offline = np.array(res_list)
data_new_offline = np.transpose(data_new_offline,[0,2,1])
print(data_new_offline.shape)
np.save(test_dataset_path/"offline_data.npy",data_new_offline)
