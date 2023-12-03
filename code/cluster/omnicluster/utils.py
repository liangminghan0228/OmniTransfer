import csv
import os
from time import time
import numpy as np
from sklearn import metrics

def cal_sse(distance_matrix: np.array, pred: np.array):
    pred_set = set(pred.tolist())
    sse_sum = 0
    for l in pred_set:
        index_list = np.where(pred==l)[0]
        if index_list.shape[0] == 1:

            continue
        sub_distance_matrix = distance_matrix[index_list,:][:, index_list]

        cluster_center = np.argmin(np.sum(sub_distance_matrix, axis=0))

        for index, distance in enumerate(sub_distance_matrix[cluster_center]):
            if index != cluster_center:
                sse_sum += pow(distance, 2)
    return sse_sum

def list2str(l):
    res = ''
    for i in l:
        res += str(i)
    return res

exp_num = 1
def time_consuming(period):
    def real_time_consuming(func):
        def cal(*args):
            start = time()
            for i in range(exp_num):
                res = func(*args)
            print(period, "time costing:", (time() - start)/exp_num, "s \n")
            return res

        return cal

    return real_time_consuming


def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a += 1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b += 1
            else:
                pass
    ri = (a + b) / (n * (n - 1) / 2)
    return ri


def get_metrics(y_true, y_pred):
    ri = rand_index(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)  # -1~1 1
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)  # -1~1 1
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)  # -1~1 1
    h = metrics.homogeneity_score(y_true, y_pred)
    c = metrics.completeness_score(y_true, y_pred)
    v = metrics.v_measure_score(y_true, y_pred)  # 0-1 1
    fmi = metrics.fowlkes_mallows_score(y_true, y_pred)  # 0-1 1
    return {
        "RI": ri,
        "ARI": ari,
        "NMI": nmi,
        "AMI": ami,
        "H": h,
        "C": c,
        "V": v,
        "FMI": fmi,
    }


def cal_save_metrics(y_true, y_pred, file_path, period="", clear=False):
    if clear and os.path.exists(file_path):
        os.remove(file_path)
    metrics_ = get_metrics(y_true, y_pred)
    with open(file_path, "a") as w:
        w.write(period)
        for metrics_name, metrics_value in metrics_.items():
            w.write(metrics_name + ":" + str(metrics_value) + "\n")
        w.write("\n")


def analyze(n_label, n_cluster, label_file, pred_file, save_file):
    label = np.load(label_file)
    pred = np.load(pred_file)
    sta_data = [([0] * n_cluster) for _ in range(n_label)]  # shape:(label,predict)
    c = [([0] * n_cluster) for _ in range(n_label)]
    r = [([0] * n_cluster) for _ in range(n_label)]
    for i_label in range(1, n_label + 1):
        _pred = pred[np.where(label == i_label)]
        for n_cluster in range(n_cluster):
            sta_data[i_label - 1][n_cluster] = np.where(_pred == n_cluster)[0].shape[0]
    c_all = np.sum(sta_data, axis=1)
    r_all = np.sum(sta_data, axis=0)
    for i_, c_data in enumerate(sta_data):
        for j_ in range(c_data.shape[0]):
            if int(c_all[i_]) == 0:
                c[i_][j_] = 0
            else:
                c[i_][j_] = sta_data[i_][j_] / int(c_all[i_])
            if int(r_all[j_]) == 0:
                r[i_][j_] = 0
            else:
                r[i_][j_] = sta_data[i_][j_] / int(r_all[j_])

    result_file = open(save_file, "w")
    writer = csv.writer(result_file)
    writer.writerow(["n_cluster:" + str(n_cluster) + "; n_label"] + str(n_label))
    first_row = []
    first_row.append("label/preidct:")
    for i_ in range(n_cluster):
        first_row.append(i_)
    write_ana(writer, first_row, sta_data)
    write_ana(writer, first_row, c)
    write_ana(writer, first_row, r)
    result_file.close()
