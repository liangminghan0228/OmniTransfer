import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from data_config import *

fig, axs = plt.subplots(len(clusters), 1, figsize=(30, 100))
train_types = ['finetune_all','finetune_freeze_0123_']
for train_type in train_types:
    exp_key = f'alpha_{global_alpha}_beta_{global_beta}_{train_type}_ws{global_window_size}{extra_info}_data1'
    exp_dir = project_path / 'out' / exp_key
    bf_df = pd.read_csv(exp_dir/f'evaluation_result/bf_machine_best_f1.csv')
    machine_index_list = bf_df['machine_id'].values
    f1_list = bf_df['f1'].values
    f1_for_each_cluster = {}
    for mi, f1 in zip(machine_index_list, f1_list):
        f1_for_each_cluster[int(mi)] = f1

    for cluster in clusters:
        machine_index_per_cluster = [str(i) for i in cluster['test']]
        f1_list_per_cluster = []
        for mi in machine_index_per_cluster:
            f1_list_per_cluster.append(f1_for_each_cluster[int(mi)])
        
        axs[int(cluster['label'])].scatter(machine_index_per_cluster, f1_list_per_cluster, label = train_type)
        axs[int(cluster['label'])].legend()

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.savefig('data1.png', bbox_inches="tight")
plt.close(fig)