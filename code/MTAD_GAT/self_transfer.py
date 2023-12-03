import json
import numpy as np
import pandas as pd
import pathlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

space = 200
data_num = 200
# prefix = "_不加权"
prefix = ""
dataset = "yidong_200实例25指标_1108_FCFW"+prefix
# out_dir = "out_1011"
out_dir = "out_1024"
model_dim = 200
is_cluster = True
# is_cluster = False
data_type = 'OC'

# json_path = f"/home/zhangshenglin/liangminghan/code/Compared_zhujun/exp_data/dataset/新数据集/OC/{dataset}/cluster_{'聚类' if is_cluster else '不聚类_vae'}.json"
use_center_dir_path = f"/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0817/data2/OC_200实例_1023_欧氏加权/use_center_不加权_5nodes_1iwi_-3.0clip_2021_1daytrain_10epoch_60ws_0.001lr"

finetunue_all_path = f"/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0817/data2/OC_200实例_1023_欧氏加权/finetune_all_不加权_5nodes_1iwi_-3.0clip_2021_1daytrain_20epoch_60ws_0.0001lr/evaluation_result/bf_machine_best_f1.csv"
freeze_rnn_path = f"/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0817/data2/OC_200实例_1023_欧氏加权/freeze_rnn_init_dense_2step_不加权_5nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr/evaluation_result/bf_machine_best_f1.csv"
# freeze_rnn_path = f"/home/zhangshenglin/liangminghan/code/Compared_zhujun/OA_pytorch/out_1024/data2/OC_200实例_1024_欧氏加权/freeze_rnn_init_dense_std_2step_不加权_5nodes_1iwi_0.01clip_2021_1daytrain_20epoch_60ws_0.001lr_200model_dim/evaluation_result/bf_machine_best_f1.csv"




if data_type == 'yidong':
    chosed_day = 7
    num_per_day = 96
    index_num = 25
else:
    chosed_day = 5
    num_per_day = 288
    index_num = 19

machine_id_list = list(range(data_num))
score_list = np.zeros(data_num)
for i in range(data_num):
    test_output = pickle.load(open(use_center_dir_path + '/result/{}/test_output.pkl'.format(i),'br'))
    # score = np.load(use_center_dir_path+f"/result/{i}/test_score.npy")
    score = test_output['A_Score_Global']

    score = np.concatenate([np.zeros((60,)), score], axis=0)

    # 取第七天数据
    score = score[num_per_day*chosed_day-num_per_day:num_per_day*chosed_day]
    # print(score.shape)

    score_agg = np.sum(score)
    # score_agg = np.max(np.sum(score, axis=1), axis=0)
    # score_agg = np.percentile(np.sum(score, axis=1), q=60)
    
    # score_agg = np.sum(np.sum(score, axis=1))
    # score_agg = np.sum(np.max(score, axis=1))
    # score_sum = np.sum(score, axis=1)
    # score_pvt = np.percentile(score_sum, 60)
    # score_agg = np.sum(score_sum[np.where(score_sum<=score_pvt)])
    score_list[i] = np.round(score_agg,2)


print(f"score min:{np.min(score_list)} max:{np.max(score_list)}")
score_min = int(np.min(score_list))
score_max = int(np.max(score_list))

# 11.3 生成每个实例的结果对比文件，并输出score的阈值遍历时，结果上升的实例个数和结果下降的实例个数（finetune all和freeze相比）
path_list = [
    pathlib.Path(finetunue_all_path),
    pathlib.Path(freeze_rnn_path),
    ]
df_list = []
for path in path_list:
    df_list.append(pd.read_csv(path))
    print(df_list[-1].shape)
col_index_list = [0,3,5,6]
df_merge =  pd.DataFrame()
for index, col in enumerate(col_index_list):
    for df_index, df in enumerate(df_list):
        if index == 0:
            tp = np.sum(df['TP'].values)
            fp = np.sum(df['FP'].values)
            fn = np.sum(df['FN'].values)
            p = tp / (tp+fp)
            r = tp / (tp+fn)
            f1 = 2*p*r/(p+r)
            print(f"{path_list[df_index].parent.parent.name} \t tp:{tp}\t fp:{fp}\t fn:{fn}\t p:{round(p, 4)}\t r:{round(r, 4)}\t f1:{round(f1, 4)}")
        # columns = df.columns
        # if df_merge.empty:
        #     df_merge['machine_id'] = df['machine_id']
        #     # df_merge['distance'] = dis_list[df['machine_id']]
        #     df_merge['score'] = score_list[df['machine_id']]
        # # print(df[columns[col]].shape)
        # new_col = f"{columns[col]}_{df_index}"
        # df_merge[new_col] = df[columns[col]]

# df_merge.to_csv("/home/zhangshenglin/liangminghan/code/OmniCluster/out_整理/临时/自适应生成的比较文件/compare_移动.csv", index=False)


# 11.4 根据异常分数的阈值得到按照该阈值组合两种微调策略得到的结果
def read_bf_all_result(best_df, dis_th,a ,b):
    tp = np.sum(best_df['TP'].values)
    fp = np.sum(best_df['FP'].values)
    fn = np.sum(best_df['FN'].values)
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    # print(f"{dis_th} all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)} {a} {b}")
    return f1
f = open("/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/mtad-gat-selftransfer.csv",mode='w')
f.write("score_th,f1,num1,num2\n")
max_f1 = -1
max_score_th = 0
# for score_th in tqdm(range(score_min, score_min+(score_max-score_min)//20+space, space)):
for score_th in tqdm(range(score_min, score_max, 5)):
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
    f.write(f"{score_th},{cur_res},{num1},{num2}\n") 
    if cur_res > max_f1:
        max_f1 = cur_res
        max_score_th = score_th
        print(f"cur_res:{cur_res}, max_score_th:{max_score_th}, num1:{num1}, num2:{num2}")

print(max_f1, max_score_th)
f.close()
# 读取理论上的最佳值
new_df = pd.DataFrame()
for machine_index, machine_id in enumerate(machine_id_list):
    if df_list[0]['f1'][machine_index]>df_list[1]['f1'][machine_index]:
        new_df = new_df.append(df_list[0].iloc[machine_index])
    else:
        new_df = new_df.append(df_list[1].iloc[machine_index])
best_res = read_bf_all_result(new_df, 0, 0, 0) 
print(f"理论上最佳结果", best_res)

# # 将实际的最佳结果区分为两个csv
# finetune_df = pd.DataFrame()
# freeze_df = pd.DataFrame()
# num1, num2 = 0 , 0
# for machine_index, machine_id in enumerate(machine_id_list):
#     if score_list[machine_id]<=max_score_th:
#         finetune_df = finetune_df.append(df_list[0].iloc[machine_index])
#         num1+=1
#     else:
#         freeze_df = freeze_df.append(df_list[1].iloc[machine_index])
#         num2 += 1
# save_dir = pathlib.Path(f"/home/zhangshenglin/liangminghan/code/Compared_zhujun/exp_data/dataset/自适应选择结果/{dataset}")
# save_dir.mkdir(exist_ok=True, parents=True)
# finetune_df.to_csv(save_dir/f"oa_finetune.csv")
# freeze_df.to_csv(save_dir/f"oa_freeze.csv")