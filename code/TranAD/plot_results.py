from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from data_config import *
import pandas as pd
index_num = feature_dim
score_dir = exp_dir / f'result'
pic_dir = exp_dir / f'pic'
font_size = 15

bf_res_prefix = 'bf_'

def plot(cluster_index, data_idx, res_type):
    bf_threshold = {}
    if (exp_dir/ f'evaluation_result/{bf_res_prefix}machine_best_f1_{res_type}.csv').exists():
        bf_df = pd.read_csv(exp_dir/ f'evaluation_result/{bf_res_prefix}machine_best_f1_{res_type}.csv')
        machine_id_list = bf_df['machine_id'].values
        threshold_list = bf_df['threshold'].values
        for machine_id, threshold in zip(machine_id_list, threshold_list):
            bf_threshold[machine_id]=threshold

    fig = plt.figure(figsize=(30, 60))
    pic_path = pic_dir /res_type/ f'{data_idx}.png'
    # pic_path = pic_dir /f'{data_idx}_raw.png'
    pic_path.parent.mkdir(parents=True, exist_ok=True)
    test_score = np.load(score_dir / f'{data_idx}/test_score_{res_type}.npy')
    # print(pic_path, test_score.shape)
    test_score[np.isnan(test_score)] = 0


    label_list = label[data_idx]
    recon_GD = np.load(score_dir/f'{data_idx}/recon_D.npy')
    recon_G = np.load(score_dir/f'{data_idx}/recon_G.npy')

    # get the joint score
  
    test_score_clip = np.where(test_score > single_score_th, single_score_th, test_score)
    test_score_sum = np.sum(test_score_clip, axis=-1)
    tp_fp_res = np.load(exp_dir/f'evaluation_result/{bf_res_prefix}tp_fp_res_{res_type}.npy')
    if dataset_type == 'data2':
        label_list = np.hstack([np.zeros(7*96), label_list])
        tp_fp_res = np.hstack([np.zeros(96*7), tp_fp_res[data_idx]])
    elif dataset_type == 'data1':
        label_list = np.hstack([np.zeros(5*288), label_list])
        tp_fp_res = np.hstack([np.zeros(288*5), tp_fp_res[data_idx]])
    elif dataset_type == 'ctf':
        label_list = np.hstack([np.zeros(5*2880+1), label_list]) 
        tp_fp_res = np.hstack([np.zeros(5*2880), tp_fp_res[data_idx]])
    
    tp_fp_res = tp_fp_res[-(len(test_score)):]
    label_list = label_list[-(len(test_score)):]

    train_data_item, test_data_item = get_data_by_index(dataset_type, data_idx, training_period)
    
    train_value, value_list = preprocess_meanstd(train_data_item.T, test_data_item[:, -(len(test_score)):].T)
    
   
    ax_score = fig.add_subplot(index_num + 3, 1, 1)
    ax_score.plot(range(len(label_list)), test_score_sum, color='green', linewidth=2)
    # ax_score.set_ylim(0,400)
    ax_score.tick_params(labelsize=font_size)
    if np.sum(label_list) > 0:
        anomaly_list = np.copy(test_score_sum)
        anomaly_list[np.where(label_list == 0)] = np.inf
        ax_score.scatter(range(label_list.shape[0]), anomaly_list, color='red', linewidth=1)
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
        ax.plot(range(label_list.shape[0]), value_list[:, row], color='black', linewidth=3)
        if res_type == 'd':
            ax.plot(range(label_list.shape[0]), recon_GD[:, row], color='blue', linewidth=1)
        elif res_type == 'g':
            ax.plot(range(label_list.shape[0]), recon_G[:, row], color='orange', linewidth=1)
        


        ax_anomaly_score = ax.twinx()
        ax_anomaly_score.tick_params(labelsize=font_size)
        if np.max(test_score[:,row]) < 5:
          ax_anomaly_score.set_ylim(0,5)
        if np.max(test_score[:,row]) > 400:
          ax_anomaly_score.set_ylim(0,400)
        ax_anomaly_score.plot(range(label_list.shape[0]), test_score[:, row], color='green', linewidth=1)

        anomaly_value_list = np.copy(value_list[:, row])
        anomaly_value_list[np.where(label_list == 0)[0]] = np.inf
        ax.scatter(range(label_list.shape[0]), anomaly_value_list, color='red', linewidth=4)

    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(pic_path, bbox_inches="tight")
    plt.close(fig)
    # print(f"{exp_key} pic:{data_idx}")

if __name__ == '__main__':
    cluster_data_list_zip = [(cluster['label'], cluster['test']) for cluster in clusters]
    cluster_data_zip = []
    for cluster_index, data_index_list in cluster_data_list_zip:
        for data in data_index_list:
            cluster_data_zip.append([cluster_index, data])
    plot_job = 30 if dataset_type != 'ctf' else 3
    from tqdm import tqdm
    Parallel(n_jobs=plot_job)(delayed(plot)(cluster_index, data_index, res_type='g') for cluster_index, data_index in tqdm(cluster_data_zip))
    # Parallel(n_jobs=plot_job)(delayed(plot)(cluster_index, data_index, res_type='d') for cluster_index, data_index in tqdm(cluster_data_zip))

