from cmath import exp
from platform import machine
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import csv
from joblib import Parallel, delayed
from data_config import *
import pandas as pd
index_num = feature_dim

score_dir = exp_dir / f'result'
pic_dir = exp_dir / f'pic'

font_size = 15

bf_threshold = {}
spot_threshold = {}
# bf_res_prefix = 'pf_'
bf_res_prefix = 'bf_'
if (exp_dir/ f'evaluation_result/{bf_res_prefix}machine_best_f1.csv').exists():
    bf_df = pd.read_csv(exp_dir/ f'evaluation_result/{bf_res_prefix}machine_best_f1.csv')
    machine_id_list = bf_df['machine_id'].values
    threshold_list = bf_df['threshold'].values
    for machine_id, threshold in zip(machine_id_list, threshold_list):
        bf_threshold[machine_id]=threshold

# print(f"bf_threshold:{bf_threshold}")

if (exp_dir/f"evaluation_result/pot_result/best_param.csv").exists():
    pot_df = pd.read_csv(exp_dir/f"evaluation_result/pot_result/best_param.csv")
    cluster_list = pot_df['cluster'].values
    q_list = pot_df['q'].values
    level_list = pot_df['level'].values
    for cluster_index, q, level in zip(cluster_list, q_list, level_list):
        best_cluster_pot_df = pd.read_csv(exp_dir/f"evaluation_result/pot_result/{cluster_index}/{q}_{level}_pot.csv")
        data_idx_list = best_cluster_pot_df['data_idx'].values
        threshold_list = best_cluster_pot_df['threshold'].values
        for data_idx, threshold in zip(data_idx_list, threshold_list):
            spot_threshold[data_idx] = -threshold


def plot(cluster_index, data_idx):
    fig = plt.figure(figsize=(30, 60))
    pic_path = pic_dir /f'{data_idx}.png'
    pic_path.parent.mkdir(parents=True, exist_ok=True)
    test_score = np.load(score_dir / f'{data_idx}/test_score.npy')
    test_score[np.isnan(test_score)] = 0


    label_list = label[data_idx]
    recon_mean = np.load(score_dir/f'{data_idx}/recon_mean.npy')
    recon_std = np.load(score_dir/f'{data_idx}/recon_std.npy')

    # get the joint score
    # test_score_clip = np.where(test_score > single_score_th, single_score_th, test_score)
    test_score_sum = np.sum(test_score, axis=-1)
    tp_fp_res = np.load(exp_dir/f'evaluation_result/{bf_res_prefix}tp_fp_res.npy')
    if dataset_type == 'data2':
        label_list = np.hstack([np.zeros(7*96), label_list])
        tp_fp_res = np.hstack([np.zeros(96*7), tp_fp_res[data_idx]])
    elif dataset_type == 'data1':
        label_list = np.hstack([np.zeros(5*288), label_list])
        tp_fp_res = np.hstack([np.zeros(288*5), tp_fp_res[data_idx]])
    elif dataset_type == 'ctf':
        label_list = np.hstack([np.zeros(5*2880+1), label_list]) 
        tp_fp_res = np.hstack([np.zeros(5*2880), tp_fp_res[data_idx]])

    label_list = label_list[-(len(test_score)):]

    
    if dataset_type == 'data2':
        preprocess_train_days = 7       
    else:
        preprocess_train_days = 5
    train_data_item, test_data_item = get_data_by_index(dataset_type, data_idx, preprocess_train_days)
    train_value, value_list = preprocess_meanstd(train_data_item.T, test_data_item[:, -(len(test_score)):].T)

    train_id_list = []
    for cluster in clusters:
        if cluster['label'] == cluster_index:
            train_id_list = cluster['train'][:15]
            break
    offline_data_list = get_offline_data(dataset_name=dataset_type, data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T)
        preprocess_offline_data_list.append(x_train)


    ax_score = fig.add_subplot(index_num + 3, 1, 1)
    ax_score.plot(range(len(label_list)), test_score_sum, color='green', linewidth=2)
    # ax_score.set_ylim(0,400)
    ax_score.tick_params(labelsize=font_size)
    if np.sum(label_list) > 0:
        anomaly_list = np.copy(test_score_sum)
        anomaly_list[np.where(label_list == 0)] = np.inf
        ax_score.scatter(range(label_list.shape[0]), anomaly_list, color='red', linewidth=1)
    if spot_threshold:
        ax_score.hlines(spot_threshold[data_idx],  0, label_list.shape[0] - 1, colors='black', linewidth=1)
    if bf_threshold:
        ax_score.hlines(bf_threshold[data_idx], 0, label_list.shape[0] - 1, colors='purple',  linewidth=1)
    ax_score.set_ylim(np.min(test_score_sum), bf_threshold[data_idx]*5)


    ax_score2 = fig.add_subplot(index_num + 3, 1, 2)
    ax_score2.plot(range(len(label_list)), test_score_sum, color='green', linewidth=1)
    # ax_score2.set_ylim(0,400)
    ax_score2.tick_params(labelsize=font_size)
    if np.sum(label_list) > 0:
        anomaly_list = np.copy(test_score_sum)
        anomaly_list[np.where(label_list == 0)] = np.inf
        ax_score2.scatter(range(label_list.shape[0]), anomaly_list, color='red', linewidth=1)
    if spot_threshold:
        ax_score2.hlines(spot_threshold[data_idx],  0, label_list.shape[0] - 1, colors='black', linewidth=1)
    if bf_threshold:
        ax_score2.hlines(bf_threshold[data_idx], 0, label_list.shape[0] - 1, colors='purple',  linewidth=1)
    
    ax_fp = fig.add_subplot(index_num + 3, 1, 3)

    tp_index = np.where(tp_fp_res==1)[0]
    fp_index = np.where(tp_fp_res==2)[0]
    fn_index = np.where(tp_fp_res==3)[0]
    tn_index = np.where(tp_fp_res==0)[0]
    # print(tn_index.shape, tp_index.shape)
    ax_fp.scatter(tp_index, np.ones_like(tp_index), linewidths=1)
    ax_fp.scatter(fp_index, 2*np.ones_like(fp_index), linewidths=1)
    ax_fp.scatter(fn_index, 3*np.ones_like(fn_index), linewidths=1)
    ax_fp.scatter(tn_index, np.zeros_like(tn_index), linewidths=1)
    ax_fp.set_ylim(-1, 4)
    ax_fp.set_ylabel(f"fp", fontsize=font_size)


    for row in range(index_num):
        ax = fig.add_subplot(index_num + 3, 1, index_num+3 - row)
        ax.tick_params(labelsize=font_size)
        ax.set_ylabel(f"{row}", fontsize=font_size)
        ax.plot(range(label_list.shape[0]), value_list[:, row], color='black', linewidth=1)
        ax.plot(range(label_list.shape[0]), recon_mean[:, row], color='blue', linewidth=1)
        ax.plot(range(label_list.shape[0]), recon_mean[:, row] + recon_std[:, row], color='orange', linewidth=1)
        ax.plot(range(label_list.shape[0]), recon_mean[:, row] - recon_std[:, row], color='orange', linewidth=1)
        for preprocess_offline_data in preprocess_offline_data_list:
            if dataset_type == 'data2':
                ax.plot(range(96*6,96*7),preprocess_offline_data[:,row])
            else:
                ax.plot(range(288*4,288*5),preprocess_offline_data[:,row])


        ax_anomaly_score = ax.twinx()
        ax_anomaly_score.tick_params(labelsize=font_size)
        if np.max(test_score[:,row]) < 30:
          ax_anomaly_score.set_ylim(0,30)
        if np.max(test_score[:,row]) > 400:
          ax_anomaly_score.set_ylim(0,400)
        ax_anomaly_score.plot(range(label_list.shape[0]), test_score[:, row], color='green', linewidth=2)

   
        anomaly_value_list = np.copy(value_list[:, row])
        anomaly_value_list[np.where(label_list == 0)[0]] = np.inf
        ax.scatter(range(label_list.shape[0]), anomaly_value_list, color='red', linewidth=1)


    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(pic_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{exp_key} pic:{data_idx}")

if __name__ == '__main__':
    cluster_data_list_zip = [(cluster['label'], cluster['test']) for cluster in clusters]
    cluster_data_zip = []
    for cluster_index, data_index_list in cluster_data_list_zip:
        for data in data_index_list:
            cluster_data_zip.append([cluster_index, data])
    plot_job = 30 if dataset_type != 'ctf' else 3
    Parallel(n_jobs=plot_job)(delayed(plot)(cluster_index, data_index) for cluster_index, data_index in cluster_data_zip)

