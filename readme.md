0.Installation

* Python == 3.7.13
* pip install -r requirements.txt
* Notices: when testing GDN, the environment configuration is as flowing
  * Python ==3.6.13
  * pip install -r requirements_gdn.txt

1.MTS clustering

```
python code/cluster/cluster.py --data_type=[model_name(data1 or data2)]

eg:python code/cluster/cluster.py --data_type=data1

```

2.Anomaly Detection

```
./run.sh [model_name] [data_type] [out_dir]
# eg: ./run.sh TranAD data2 1224
```

3.adaptive transfer

Modify the parameters passed in by the adaptive transfer script **(data_type, use_center_dir_path, finetunue_all_path, freeze_rnn_path)**  to get the final results.

```
python code/transfer_eval.py --data_type=[data_type] --model_name=[model_name] --use_center_dir_path=[use_center_file_dir_path] --finetunue_all_path=[finetune_all_csv_path] --freeze_init_path=[freeze_init_csv_path]

#eg:  code/python transfer_eval.py --data_type=data2 --use_center_dir_path=1011/TranAD/data2/use_center_20nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.95eps --finetunue_all_path=1011/TranAD/data2/finetune_all_20nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/bf_machine_best_f1_g.csv --freeze_init_path=1011/TranAD/data2/freeze_att_init_last_2step_20nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_20epoch_256bs_60ws_0.95eps/evaluation_result/bf_machine_best_f1_g.csv --model_name=TranAD
```
