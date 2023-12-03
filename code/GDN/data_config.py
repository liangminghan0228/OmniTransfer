import pathlib
import json
import numpy as np
import os
import argparse
import torch
project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--topk', type=int, default=20)
parser.add_argument('--window_size', type=int, default=60)

# training
parser.add_argument('--epochs', type=int, default= 10)
parser.add_argument('--index_weight', type=int, default= 10)
parser.add_argument('--lr', type=float, default= 1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_epoch_freq', type=int, default=5) 

parser.add_argument('--dataset_path',type=str)
parser.add_argument('--train_num',type=int,default=5)
parser.add_argument('--index_weight_index',type=int,default=1)

args = parser.parse_args()

single_score_th = 10000
out_dir = args.out_dir

GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args.dataset_type
global_epochs = args.epochs
seed = args.seed
training_period = args.training_period

global_embed_dim= args.embed_dim
global_batch_size= args.batch_size
global_out_layer_num = 1
global_out_layer_inter_dim = 256
global_topk = args.topk
# learning rate
global_learning_rate = args.lr
circle_loss_weight = args.index_weight

# global_min_std = args.min_std
dataset_path = args.dataset_path
index_weight_index = args.index_weight_index
train_num = args.train_num
if_freeze_seq = True
train_type = args.train_type

exp_key = train_type
exp_key += f"_{train_num}nodes"
exp_key += f"_{index_weight_index}iwi"
# exp_key += f"_{global_min_std}clip"
exp_key += f"_{seed}"
exp_key +=f"_{training_period}daytrain"
exp_key += f"_{global_epochs}epoch"
exp_key += f"_{args.window_size}ws"
exp_key += f"_{args.lr}lr"
exp_dir = pathlib.Path(out_dir) /"GDN" / dataset_type / exp_key

if 'initr' in exp_key:
    global_learning_rate = 5e-3
base_model_dir = args.base_model_dir


learning_rate_decay_by_epoch = 10

learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch_freq

dataset_root = pathlib.Path(f"code/test_dataset/{dataset_type}")
online_data_path = dataset_root/"online_data.npy"
offline_data_path = dataset_root / "offline_data.npy"
label_path = dataset_root/"label.npy"
cluster_json_path = dataset_root/'cluster.json'



def wrap_data(data, window_size):
    data = np.concatenate([data[-(window_size-1):], data], axis=0)
    return data


def get_data_by_index(dataset_name, data_index, training_period):
    
    if dataset_name == 'data2':
        data = np.load(online_data_path)
        train_data = data[:, :, 672-96*training_period:672]
        test_data = data[:, :, :]
        return train_data[data_index], test_data[data_index]
    
    elif dataset_name == 'data1':
        data = np.load(online_data_path)
        test_data = data
        train_data = data[:, :, 1440-288*training_period:1440]
        return train_data[data_index], test_data[data_index]
    
    elif dataset_name == 'ctf':
        data = np.load(project_path.parent / f'exp_data/data_raw/CTF_dataset/CTF_data_533_npy/{data_index}.npy')
        try:
            return data[:, int(2880*5-2880*training_period):2880*5], data[:, 2880*5:]
        except:
            return data[:, :2880*5], data[:, 2880*5:]

def get_offline_data(dataset_name, data_index_list):
    if dataset_name == 'data2':
        data = np.load(offline_data_path)
        return [data[index, :, :] for index in data_index_list]
    elif dataset_name == 'data1':
        data = np.load(offline_data_path)
        return [data[index, :, :] for index in data_index_list]
    elif dataset_name == 'ctf':
        # data = np.load(project_path.parent / 'exp_data/anotransfer_data/CTF/offline_data_2day_train.npy')
        data = np.load(project_path.parent / 'exp_data/anotransfer_data/CTF_all/offline_data_2day_train.npy')
        return [data[index, :, :] for index in data_index_list]

def get_recon_index_weight(dataset_type):
    index_weight = np.ones(np.load(online_data_path).shape[1])  
    print(index_weight)
    return index_weight

index_loss_weight = get_recon_index_weight(dataset_type)


if dataset_type == 'data2':
    day_points = 96
    # min_std = global_min_std
    # min_std = 0
    global_window_size = args.window_size
    feature_dim = 25
    # feature_dim = 38
    label = np.load(label_path)

    cluster_json_path = cluster_json_path      
    bf_search_min = 0
    bf_search_max = 1000
    bf_search_step_size = 1
    eval_item_length = 96*7 - (global_window_size - 1)
    noshare_save_dir = project_path / base_model_dir

elif dataset_type == 'data1':
    day_points = 288
    # min_std = global_min_std
    feature_dim = 19
    # feature_dim = 18
    global_window_size = args.window_size
    label = np.load(label_path)

    cluster_json_path = cluster_json_path   
    bf_search_min = 0
    bf_search_max = 1000
    bf_search_step_size = 1
    noshare_save_dir = project_path / base_model_dir

    eval_item_length = 288*2 - (global_window_size - 1)

elif dataset_type == 'ctf':
    # global_valid_step_freq = 3 if ('freeze' in exp_key or 'finetune' in exp_key) else 20
    feature_dim = 49
    index_loss_weight = np.array([1 for _ in range(feature_dim)])
    ctf_chose_index = [0, 1, 2, 4, 7, 10, 12, 13, 14, 32, 33, 36, 38]

    index_loss_weight[ctf_chose_index] = circle_loss_weight

    noshare_save_dir = project_path / base_model_dir

    global_window_size = 15
    # feature_dim = 13
    label = np.load(project_path.parent / "exp_data/data_raw/CTF_dataset/ctf_label.npy")
    
    cluster_json_path = project_path.parent / 'exp_data/config/cluster_ctf_ano_meanstd_bk_18instance.json'

    # best f1score threshold
    bf_search_min = 0
    bf_search_max = 10000
    bf_search_step_size = (bf_search_max - bf_search_min) // 1000
    eval_item_length = 2880*8 - (global_window_size - 1)

data_list = []
with open(cluster_json_path, 'r') as cl:
    cs = json.load(cl)
for c in cs:
    data_list.extend(c['test'])
# data_list = data_list[:1]
# print('data_list:', data_list)
eval_data_list = data_list
if 'noshare' not in exp_key:
    clusters = cs
else:
    clusters = [{"label": i, "center": i, "train": [i], "test":[i]} for i in data_list]


from sklearn.preprocessing import MinMaxScaler
def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """
    # print('meanstd', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    k = 3
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[np.where(std_array==0)] = e
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    
    train_mean_array = np.mean(df_train, axis=0, keepdims=True)
    train_std_array = np.std(df_train, axis=0, keepdims=True)
    train_std_array[np.where(train_std_array==0)] = e
    
    df_train_new = (df_train - train_mean_array) / train_std_array
    
    df_test = np.where(df_test > train_mean_array + k * train_std_array, train_mean_array + k * train_std_array, df_test)
    df_test = np.where(df_test < train_mean_array - k * train_std_array, train_mean_array - k * train_std_array, df_test)
    df_test_new = (df_test - train_mean_array) / train_std_array

    return df_train_new, df_test_new