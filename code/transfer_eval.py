import json
import numpy as np
import pandas as pd
import pathlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_center_dir_path', type=str)
parser.add_argument('--finetunue_all_path', type=str)
parser.add_argument('--freeze_init_path', type=str)
parser.add_argument('--data_type', type=str)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
data_type = args.data_type
use_center_dir_path = args.use_center_dir_path
finetunue_all_path = args.finetunue_all_path
freeze_init_path = args.freeze_init_path
model_name=args.model_name

data_num = 200
# with open("code/test_dataset/data2/cluster.json") as f:
#     clusters=json.load(f)
#     for cluster in clusters:
#         data_num+=len(cluster['test'])

if data_type == 'data2':
    chosed_day = 7
    num_per_day = 96
else:
    chosed_day = 5
    num_per_day = 288

machine_id_list = list(range(data_num))
score_list = np.zeros(data_num)
for i in range(data_num):
    if model_name=="TranAD" or model_name=="USAD":
        score = np.load(use_center_dir_path+f"/result/{i}/test_score_g.npy")
    else:
        score = np.load(use_center_dir_path+f"/result/{i}/test_score.npy")

    score = score[num_per_day*chosed_day-num_per_day:num_per_day*chosed_day]
    # print(score.shape)
    # score_agg = np.max(np.sum(score, axis=1), axis=0)
    # score_agg = np.percentile(np.sum(score, axis=1), q=60)
    
    # score_agg = np.sum(np.sum(score, axis=1))
    score_agg = np.sum(score)
    # score_sum = np.sum(score, axis=1)
    # score_pvt = np.percentile(score_sum, 60)
    # score_agg = np.sum(score_sum[np.where(score_sum<=score_pvt)])
    score_list[i] = np.round(score_agg,2)
print(f"score min:{np.min(score_list)} max:{np.max(score_list)}")
score_min = int(np.min(score_list))
score_max = int(np.max(score_list))

path_list = [
    pathlib.Path(finetunue_all_path),
    pathlib.Path(freeze_init_path),
    ]
df_list = []
for path in path_list:
    df_list.append(pd.read_csv(path))
    print(df_list[-1].shape)
col_index_list = [6,1,2,3]

df_merge =  pd.DataFrame()
for index, col in enumerate(col_index_list):
    for df_index, df in enumerate(df_list):
        if index == 0:
            tp = np.sum(df['tp'].values)
            fp = np.sum(df['fp'].values)
            fn = np.sum(df['fn'].values)
            p = tp / (tp+fp)
            r = tp / (tp+fn)
            f1 = 2*p*r/(p+r)
            print(f"{path_list[df_index].parent.parent.name} \t tp:{tp}\t fp:{fp}\t fn:{fn}\t p:{round(p, 4)}\t r:{round(r, 4)}\t f1:{round(f1, 4)}")
        columns = df.columns
        if df_merge.empty:
            df_merge['machine_id'] = df['machine_id']
            # df_merge['distance'] = dis_list[df['machine_id']]
            df_merge['score'] = score_list[df['machine_id']]
        # print(df[columns[col]].shape)
        new_col = f"{columns[col]}_{df_index}"
        df_merge[new_col] = df[columns[col]]


def read_bf_all_result(best_df, dis_th,a ,b):
    tp = np.sum(best_df['tp'].values)
    fp = np.sum(best_df['fp'].values)
    fn = np.sum(best_df['fn'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    # print(f"{dis_th} all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)} {a} {b}")
    return f1


max_f1 = -1
max_score_th = 0
for score_th in tqdm(range(score_min, score_max+101, (score_max+101-score_min)//100)):
    new_df = pd.DataFrame()
    num1, num2 = 0 , 0
    for machine_index, machine_id in enumerate(machine_id_list):
        if score_list[machine_id]<=score_th:
            new_df = new_df.append(df_list[0].iloc[machine_index])
            num1+=1
        else:
            new_df = new_df.append(df_list[1].iloc[machine_index])
            num2 += 1
    cur_res = read_bf_all_result(new_df, score_th, num1, num2)

    if cur_res > max_f1:
        max_f1 = cur_res
        max_score_th = score_th
        print(f"cur_res:{cur_res}, max_score_th:{max_score_th}, num1:{num1}, num2:{num2}")

print(max_f1, max_score_th)

