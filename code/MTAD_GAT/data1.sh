# export CUDA_VISIBLE_DEVICES=1


python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/example_finetune.py --lr=1e-4 --dataset_type='yidong' --train_type='finetune_all_不加权' --epochs=20 --training_period=6 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --index_weight_index=1 --out_dir=out_0818  --min_std=-3
python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/example_finetune2step.py --lr=1e-3 --dataset_type='yidong' --train_type='freeze_rnn_init_dense_2step_不加权' --epochs=30 --training_period=6 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --out_dir=out_0818  --min_std=-3

python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/example_finetune.py --lr=1e-4 --dataset_type='yidong' --train_type='finetune_all_不加权' --epochs=20 --training_period=7 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --index_weight_index=1 --out_dir=out_0818  --min_std=-3
python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/example_finetune2step.py --lr=1e-3 --dataset_type='yidong' --train_type='freeze_rnn_init_dense_2step_不加权' --epochs=30 --training_period=7 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --out_dir=out_0818  --min_std=-3




python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/get_eval_result.py --lr=1e-4 --dataset_type='yidong' --train_type='finetune_all_不加权' --epochs=20 --training_period=6 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --index_weight_index=1 --out_dir=out_0818  --min_std=-3
python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/get_eval_result.py --lr=1e-3 --dataset_type='yidong' --train_type='freeze_rnn_init_dense_2step_不加权' --epochs=30 --training_period=6 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --out_dir=out_0818  --min_std=-3

python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/get_eval_result.py --lr=1e-4 --dataset_type='yidong' --train_type='finetune_all_不加权' --epochs=20 --training_period=7 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --index_weight_index=1 --out_dir=out_0818  --min_std=-3
python /home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/get_eval_result.py --lr=1e-3 --dataset_type='yidong' --train_type='freeze_rnn_init_dense_2step_不加权' --epochs=30 --training_period=7 --seed=2021 --gpu_id=1  --base_model_dir=/home/zhangshenglin/chezeyu/MTS_AD/MTAD_GAT/out_0818/yidong/yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值/offline_不加权_20nodes_1iwi_-3.0clip_2021_1daytrain_30epoch_60ws_0.001lr --dataset_path=yidong_200实例25指标_1108_欧氏加权_平滑5_去除极值 --train_num=20  --out_dir=out_0818  --min_std=-3


#yidong

