from re import I
import re

from numpy.core.defchararray import index
import configs
from utils import *
import numpy as np
import pandas as pd
import json
from collections import Counter
import sklearn.metrics as sm
import sklearn.mixture as smix
import cluster_kmeans
CONFIG_PATH = 'configs'



# @time_consuming('cluster time')
def main(params, remove_mask_label, cluster_approach):
    # label = np.load(params["label_file"])
    # remove_mask = np.where(label == remove_mask_label)[0]
    # print(f"remove_mask.shape {remove_mask.shape}")
    # label = np.delete(label, remove_mask)
    for dis in params["dis_list"]:
        distance_matrix = np.load(
            params["distance_dir"].joinpath(f"{dis}.npy"))
        # sub_distance_matrix = distance_matrix

        pred = None
        if cluster_approach == 'AgglomerativeClustering':
            # AgglomerativeClustering
            cengci_start = time()
            pred = sc.AgglomerativeClustering(n_clusters=None, compute_full_tree=True, affinity="precomputed",
                                              linkage=params["linkage"],
                                              distance_threshold=params["distance_threshold"]).fit_predict(distance_matrix)
            cengci_end = time()
            print(f"AgglomerativeClusteringTime:{cengci_end-cengci_start}")
        elif cluster_approach == 'DBSCAN':
            # DBSCAN
            # pred = sc.DBSCAN(eps=params["distance_threshold"], min_samples=params['sample_threshold'], metric='precomputed', algorithm=params['dbscan_algorithm']).fit_predict(distance_matrix)
            dbscan_start = time()
            pred = sc.DBSCAN(eps=params["distance_threshold"], min_samples=params['sample_threshold'],
                             metric='precomputed').fit_predict(distance_matrix)
            dbscan_end = time()
            print(f"dbscan time:{dbscan_end-dbscan_start}")
        elif cluster_approach == 'kmeans':
            kmeans_start = time()
            pred = cluster_kmeans.main(np.load(params['z_to_cluster']), int(params["distance_threshold"]))
            print(f"kmeans time:{time()-kmeans_start}")
        else:
            raise Exception

        pred_num = len(set(pred.tolist()))
        print(pred_num)
        np.save(params['pred_dir'] /
                f'cluster_pred_{params["distance_threshold"]}_{cluster_approach}.npy', pred)
        index = np.array(list(range(1, pred.shape[0]+1)))
        np.savetxt(params['pred_dir'] / f'cluster_pred_{params["distance_threshold"]}_{cluster_approach}.txt', np.transpose(np.vstack([index, pred])), fmt='%d', delimiter=' : ')
        

        for label_index in range(pred_num+1):
            if np.count_nonzero(pred == label_index) == 1:
                pred[np.where(pred == label_index)[0]] = -1

        pred_num = len(set(pred.tolist()))

        np.save(params['pred_dir'] / f'cluster_pred_{params["distance_threshold"]}_{cluster_approach}_after.npy', pred)
        index = np.array(list(range(1, pred.shape[0]+1)))
        np.savetxt(params['pred_dir'] / f'cluster_pred_{params["distance_threshold"]}_{cluster_approach}_after.txt', np.transpose(np.vstack([index, pred])), fmt='%d', delimiter=' : ')


if __name__ == '__main__':
    config = configs.config(CONFIG_PATH)
    paras = {
        "dis_list": json.loads(config.get('cluster', 'dis_list')),
        "distance_dir": pathlib.Path(config.get('cal_dis', 'distance_dir')),
        "linkage": config.get('cluster', 'linkage'),
        "distance_threshold": config.getfloat('cluster', 'distance_threshold'),
        "label_file": config.get('cluster', 'label_file'),
        "result_dir": pathlib.Path(config.get('cluster', 'result_dir')),
        'pred_dir': pathlib.Path('/home/liangminghan/code/DeepClustering/new-data/')
    }
    for train_param in configs.train_param_list:
        name_key = f"layers_num_{train_param['layers_num']}_kernel_size_list_{list2str(train_param['kernel_size_list'])}_stride_list_{list2str(train_param['stride_list'])}_output_padding_list_{list2str(train_param['output_padding_list'])}_filters_list_{list2str(train_param['filters_list'])}_dropout_list_{list2str(train_param['dropout_list'])}_batch_size_{config.getint('train_ae', 'batch_size')}"

        distance_dir = pathlib.Path(config.get('cal_dis', 'distance_dir'))
        distance_dir = distance_dir / name_key
        result_dir = pathlib.Path(config.get('cluster', 'result_dir'))
        result_dir = result_dir / name_key

        paras['result_dir'] = result_dir
        paras['distance_dir'] = distance_dir
        paras["result_dir"].mkdir(parents=True, exist_ok=True)

        for dis_th_index in range(6, 7, 1):
            dist_th = dis_th_index
            paras['distance_threshold'] = dist_th
            samples_th = 1
            paras['sample_threshold'] = samples_th
            metrics = main(paras, 73)
