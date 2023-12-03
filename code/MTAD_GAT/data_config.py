import pathlib
import json
import numpy as np
import os
import argparse
import torch
from model.utils import *

project_path = pathlib.Path(os.path.abspath(__file__)).parent

parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--base_model_dir', type=str, default=None)
parser.add_argument('--window_size', type=int, default=60)

# -- Model params ---
# 1D conv layer
parser.add_argument("--kernel_size", type=int, default=7)
# GAT layers
parser.add_argument("--use_gatv2", type=str2bool, default=True)
parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
parser.add_argument("--time_gat_embed_dim", type=int, default=None)
# GRU layer
parser.add_argument("--gru_n_layers", type=int, default=1)
parser.add_argument("--gru_hid_dim", type=int, default=150)
# Forecasting Model
parser.add_argument("--fc_n_layers", type=int, default=3)
parser.add_argument("--fc_hid_dim", type=int, default=150)
# Reconstruction Model
parser.add_argument("--recon_n_layers", type=int, default=1)
parser.add_argument("--recon_hid_dim", type=int, default=150)
# Other
parser.add_argument("--alpha", type=float, default=0.2)

# --- Predictor params ---
parser.add_argument("--scale_scores", type=str2bool, default=False)
parser.add_argument("--use_mov_av", type=str2bool, default=False)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--level", type=float, default=None)
parser.add_argument("--q", type=float, default=None)
parser.add_argument("--dynamic_pot", type=str2bool, default=False)



# training
parser.add_argument('--epochs', type=int, default= 10)
parser.add_argument('--index_weight', type=int, default= 10)
parser.add_argument('--lr', type=float, default= 1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--log_tensorboard", type=str2bool, default=True)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
parser.add_argument("--dropout", type=float, default=0.3)


parser.add_argument('--train_type', type=str)
parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_epoch', type=int, default=5) 

parser.add_argument('--min_std', type=float, default=0) 
parser.add_argument('--dataset_path',type=str)
parser.add_argument('--train_num',type=int,default=5)
parser.add_argument('--index_weight_index',type=int,default=1)

parser.add_argument('--use_cluster_json_path',type=str,default='')

parser.add_argument('--beta',type=str,default='')

args = parser.parse_args()
args_summary = str(args.__dict__)
use_cluster_json_path = args.use_cluster_json_path
beta = args.beta

out_dir = args.out_dir
GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')

dataset_type = args.dataset_type
global_epochs = args.epochs
seed = args.seed
training_period = args.training_period


global_batch_size= args.batch_size
global_learning_rate = args.lr
circle_loss_weight = args.index_weight
train_type = args.train_type
train_num = args.train_num
if_freeze_seq = True # 固定序列化的部分的参数，模型默认固定的，不固定的话很容易训练崩溃

global_min_std = args.min_std
dataset_path = args.dataset_path
index_weight_index = args.index_weight_index

exp_key = train_type
exp_key += f"_{train_num}nodes"
exp_key += f"_{index_weight_index}iwi"
exp_key += f"_{global_min_std}clip"
exp_key += f"_{seed}"
exp_key +=f"_{training_period}daytrain"
exp_key += f"_{global_epochs}epoch"
exp_key += f"_{args.window_size}ws"
exp_key += f"_{args.lr}lr"

base_model_dir = args.base_model_dir
if beta!='':
    exp_key += f'_beta{beta}'
    base_model_dir += f'_beta{beta}'

exp_dir = project_path / out_dir / dataset_type /dataset_path/ exp_key


# 学习率衰减
learning_rate_decay_by_epoch = 10
learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch


dataset_root = project_path.parent / "exp_data/dataset/新数据集" / f"{'OC' if dataset_type=='data2' else 'yidong'}"/dataset_path
online_data_path = dataset_root/"online_data.npy"
# online_data_path = "/home/zhangshenglin/liangminghan/code/OmniCluster/out_整理/注入异常/online_data_加入异常.npy"
offline_data_path = dataset_root / "offline_data.npy"
cmndf_path = dataset_root/"cmndf.npy"
label_path = dataset_root/"label.npy"
cluster_json_path = dataset_root/'cluster_聚类.json' if use_cluster_json_path=='' else use_cluster_json_path
cluster_json_path_不聚类 = dataset_root/'cluster_不聚类_vae.json'
if beta!='':
    print('beta =',beta)
    cluster_json_path = dataset_root/f'cluster_beta{beta}.json'

print(exp_dir)
print(dataset_root)

def get_data_by_index(dataset_name, data_index, training_period):
    # 通过数据集名称，数据index，训练数据，获取某个数据集的训练集和测试集
    if dataset_name == 'yidong':
        data = np.load(online_data_path)
        train_data = data[:, :, 672-96*training_period:672]
        test_data = data[:, :, :]
        return train_data[data_index], test_data[data_index]
    
    elif dataset_name == 'data2':
        data = np.load(online_data_path)
        test_data = data
        train_data = data[:, :, 1440-288*training_period:1440]
        return train_data[data_index], test_data[data_index]

def get_offline_data(dataset_name, data_index_list):
    if dataset_name == 'yidong':
        data = np.load(offline_data_path)
        return [data[index, :, :] for index in data_index_list]
    elif dataset_name == 'data2':
        data = np.load(offline_data_path)
        return [data[index, :, :] for index in data_index_list]


# 读取YIN算法输出的指标周期性强弱
def get_recon_index_weight(dataset_type):
    if dataset_type == 'yidong':
        cmndf = np.load(cmndf_path)
        index_weight = 1 / (np.mean(cmndf, axis=0))**index_weight_index
        if "不聚类" in out_dir or "不加权" in out_dir or '不聚类' in exp_key or '不加权' in exp_key:
            index_weight = np.ones_like(index_weight)
        # index_weight = 1 / (np.mean(cmndf, axis=0))
    elif dataset_type == 'data2':
        cmndf = np.load(cmndf_path)
        index_weight = 1 / (np.mean(cmndf, axis=0))**index_weight_index
        if "不聚类" in out_dir or "不加权" in out_dir or '不聚类' in exp_key or '不加权' in exp_key:
            index_weight = np.ones_like(index_weight)   
    index_weight = np.ones_like(index_weight)   
    print(index_weight)
    return index_weight
index_loss_weight = get_recon_index_weight(dataset_type)

# 实验参数
if dataset_type == 'yidong':
    min_std = global_min_std
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
    # noshare_save_dir = project_path / base_model_dir

elif dataset_type == 'data2':
    min_std = global_min_std
    feature_dim = 19
    # feature_dim = 18
    global_window_size = args.window_size
    label = np.load(label_path)

    cluster_json_path = cluster_json_path   
    bf_search_min = 0
    bf_search_max = 1000
    bf_search_step_size = 1
    
    # noshare_save_dir = project_path / base_model_dir

    eval_item_length = 288*2 - (global_window_size - 1)

noshare_save_dir = pathlib.Path(base_model_dir)
data_list = []
with open(cluster_json_path, 'r') as cl:
    cs = json.load(cl)
for c in cs:
    data_list.extend(c['test'])  
eval_data_list = data_list
if 'noshare' not in exp_key:
    clusters = cs
    if '不聚类' in out_dir or '不聚类' in exp_key:
        with open(cluster_json_path_不聚类, 'r') as cl:
            cs = json.load(cl)
            clusters = cs

else:
    clusters = [{"label": i, "center": i, "train": [i], "test":[i]} for i in data_list]


from sklearn.preprocessing import MinMaxScaler

k = 5
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

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test


def preprocess(x_train,x_test):
    x_train, scaler = normalize_data(x_train, scaler=None)
    x_test, _ = normalize_data(x_test, scaler=scaler)
    return x_train,x_test



"""
使用requests库自动注册作业进程（更新已有作业信息）
"""
import requests
import re
import os

def login(username, password, pid,login_url='http://10.10.1.210/site/login',url_index=3527,server_user='zhangshenglin',id_server=219): #FIXME 更新作业记录的接口
    update_url="http://10.10.1.210/job/{}/update".format(url_index)
    #请求头
    my_headers = {
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.35',
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding' : 'gzip,deflate',
        'Accept-Language' : 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6'
    }
 
    #get请求获取csrf token
    session = requests.Session()
    response = session.get(login_url, headers=my_headers)
    reg = r'<meta name="csrf-token" content="(.*)">'
    pattern = re.compile(reg)
    result = pattern.findall(response.content.decode('utf-8'))
    token = result[0]
    # print(f"登录请求的token={token}")
    
    #登录post的data
    login_data = {
        '_csrf': token,
        'LoginForm[student_id]': username,
        'LoginForm[password]':password,
        'login-button': ''
    }
    
    #post请求正式登录
    response = session.post(login_url, headers=my_headers, data=login_data)
    # print(response.content.decode('utf-8'))

    #再次get请求update的网址获取csrf token
    response = session.get(update_url, headers=my_headers)
    reg = r'<meta name="csrf-token" content="(.*)">'
    pattern = re.compile(reg)
    result = pattern.findall(response.content.decode('utf-8'))
    token = result[0]
    # print(f"更新作业进程请求的token={token}")
    # input()

    index2server = {219:'15',218:'14'}
    server_id = index2server.get(id_server,0)

    #修改注册进程号post的data
    update_data = {
        '_csrf': token,
        'Job[status]': '0',
        'Job[duration]': '5',
        'Job[id_server]': server_id, #FIXME 服务器的编号
        'Job[server_user]': server_user, #FIXME 用户名
        'Job[pid]': pid, #FIXME 进程号
        'Job[comm]': 'python', #FIXME 运行的命令名字只需要第一个单词
        # 'Job[use_gpu]': '0',
        'Job[use_gpu]': '1', #FIXME 使用GPU
        'Job[description]': '实验' #FIXME 作业记录的描述
    }

    #post请求修改注册的进程号
    response = session.post(update_url, headers=my_headers, data=update_data)
    # print(response.content.decode('utf-8'))


login(username="1913060", password="20200710", pid=str(os.getpid()),url_index=3527,server_user='zhangshenglin',id_server=218)
