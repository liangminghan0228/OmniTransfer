model_name=$1
dataset_type=$2
out_dir=$3

train_num=0
if test $dataset_type = 'data2'
then
    train_num=20
elif test $dataset_type = 'data1'
then
    train_num=5
fi

if test $model_name = 'DAGMM'
then
    echo "run DAGMM..."
    # Offline Training
    python code/DAGMM/example_offline.py --dataset_type=$dataset_type --train_type='offline'  --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir --valid_epoch=25

    python code/DAGMM/example_use_center.py --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/DAGMM/$dataset_type/offline_2021_${train_num}nodes_1iwi_1daytrain_epoch500_lr0.001_dropout_r0.1_weight10_originsample1024 --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1  --index_weight_index=1 --out_dir=$out_dir
    # Transfer Learning
    python code/DAGMM/example_finetune.py --dataset_type=$dataset_type --train_type='finetune_all' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/DAGMM/$dataset_type/offline_2021_${train_num}nodes_1iwi_1daytrain_epoch500_lr0.001_dropout_r0.1_weight10_originsample1024 --lr=5e-3 --epoch=20 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1  --index_weight_index=1 --out_dir=$out_dir

    python code/DAGMM/example_finetune.py --dataset_type=$dataset_type --train_type='freeze_vae_12_init_vae_3' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/DAGMM/$dataset_type/offline_2021_${train_num}nodes_1iwi_1daytrain_epoch500_lr0.001_dropout_r0.1_weight10_originsample1024 --lr=3e-3 --epoch=50 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir
    # Online Detection
    python code/DAGMM/get_eval_result.py --dataset_type=$dataset_type --train_type='finetune_all' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/DAGMM/$dataset_type/offline_2021_${train_num}nodes_1iwi_1daytrain_epoch500_lr0.001_dropout_r0.1_weight10_originsample1024 --lr=5e-3 --epoch=20 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1  --index_weight_index=1 --out_dir=$out_dir

    python code/DAGMM/get_eval_result.py --dataset_type=$dataset_type --train_type='freeze_vae_12_init_vae_3' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/DAGMM/$dataset_type/offline_2021_${train_num}nodes_1iwi_1daytrain_epoch500_lr0.001_dropout_r0.1_weight10_originsample1024 --lr=3e-3 --epoch=50 --dropout_r=0.1 --origin_samples=1024 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir
elif test $model_name = 'GDN'
then
    echo "run GDN..."
    # Offline Training
    python code/GDN/example_offline.py --dataset_type=$dataset_type --epochs=50 --train_type='offline' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir --embed_dim=64

    python code/GDN/example_use_center.py --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/GDN/$dataset_type/offline_${train_num}nodes_1iwi_2021_1daytrain_50epoch_60ws_0.005lr --lr=5e-3 --topk=19 --train_num=$train_num --epochs=50 --index_weight_index=1 --out_dir=$out_dir --embed_dim=64
    # Transfer Learning
    python code/GDN/example_finetune.py --dataset_type=$dataset_type --train_type='finetune_all'  --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/GDN/$dataset_type/offline_${train_num}nodes_1iwi_2021_1daytrain_50epoch_60ws_0.005lr  --lr=5e-4 --topk=19  --epochs=10 --valid_epoch_freq=1 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir  --embed_dim=64

    python code/GDN/example_finetune2step.py --dataset_type=$dataset_type --train_type='freeze_init_2step'  --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/GDN/$dataset_type/offline_${train_num}nodes_1iwi_2021_1daytrain_50epoch_60ws_0.005lr  --lr=5e-3 --topk=19  --epochs=20 --valid_epoch_freq=1 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir --embed_dim=64
    # Online Detection
    python code/GDN/get_eval_result.py --dataset_type=$dataset_type --train_type='finetune_all'  --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/GDN/$dataset_type/offline_${train_num}nodes_1iwi_2021_1daytrain_50epoch_60ws_0.005lr  --lr=5e-4 --topk=19  --epochs=10 --valid_epoch_freq=1 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir  --embed_dim=64

    python code/GDN/get_eval_result.py --dataset_type=$dataset_type --train_type='freeze_init_2step'  --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=$out_dir/GDN/$dataset_type/offline_${train_num}nodes_1iwi_2021_1daytrain_50epoch_60ws_0.005lr  --lr=5e-3 --topk=19  --epochs=20 --valid_epoch_freq=1 --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir --embed_dim=64
elif test $model_name = 'InterFusion'
then
    echo "run InterFusion..."
    # Offline Training
    python code/InterFusion/example_offline.py --dataset_type=$dataset_type --train_type='offline' --epochs=10 --training_period=1 --seed=2021 --gpu_id=0 --model_dim=300 --base_model_dir=test --valid_epoch=1 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01

    python code/InterFusion/example_use_center.py --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0 --model_dim=300 --base_model_dir=$out_dir/InterFusion/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_modeldim300_epoch10_lr0.0005 --valid_epoch=1 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01
    # Transfer Learning
    python code/InterFusion/example_finetune.py --train_lr=3e-4 --dataset_type=$dataset_type --train_type='finetune_all' --epochs=10 --training_period=3 --seed=2021  --gpu_id=0 --model_dim=300 --base_model_dir=$out_dir/InterFusion/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_modeldim300_epoch10_lr0.0005 --valid_epoch=1 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01 

    python code/InterFusion/example_finetune2step.py --train_lr=5e-4  --dataset_type=$dataset_type --train_type='freeze_rnn_cnn_init_std_2step' --epochs=40  --training_period=1 --seed=2021 --gpu_id=0 --model_dim=300 --base_model_dir=$out_dir/InterFusion/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_modeldim300_epoch10_lr0.0005 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01 --valid_epoch=1
    # Online Detection
    python code/InterFusion/get_eval_result.py --train_lr=3e-4 --dataset_type=$dataset_type --train_type='finetune_all' --epochs=10 --training_period=3 --seed=2021  --gpu_id=0 --model_dim=300 --base_model_dir=$out_dir/InterFusion/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_modeldim300_epoch10_lr0.0005 --valid_epoch=1 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01 

    python code/InterFusion/get_eval_result.py --train_lr=5e-4  --dataset_type=$dataset_type --train_type='freeze_rnn_cnn_init_std_2step' --epochs=40  --training_period=1 --seed=2021 --gpu_id=0 --model_dim=300 --base_model_dir=$out_dir/InterFusion/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_modeldim300_epoch10_lr0.0005 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=0.01 --valid_epoch=1
elif test $model_name = 'OmniAnomaly'
then
    echo "run OmniAnomaly..."
    # Offline Training
    python code/OmniAnomaly/example_offline.py --dataset_type=$dataset_type --train_type='offline' --epochs=50 --training_period=1 --seed=2021 --gpu_id=0 --model_dim=200 --base_model_dir=test --train_num=$train_num --index_weight_index=1 --out_dir=$out_dir --min_std=0.01

    python code/OmniAnomaly/example_use_center.py --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0 --model_dim=200 --epoch=50 --base_model_dir=$out_dir/OmniAnomaly/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_50epoch_60ws_0.001lr_200model_dim --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=0.01
    # Transfer Learning
    python code/OmniAnomaly/example_finetune.py --dataset_type=$dataset_type --train_type='finetune_all' --epochs=10 --training_period=1 --seed=2021 --lr=5e-4 --valid_epoch=1 --gpu_id=0 --model_dim=200 --base_model_dir=$out_dir/OmniAnomaly/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_50epoch_60ws_0.001lr_200model_dim --train_num=$train_num --out_dir=$out_dir  --min_std=0.01 

    python code/OmniAnomaly/example_finetune2step.py  --dataset_type=$dataset_type --train_type='freeze_rnn_init_dense_std_2step' --training_period=1 --seed=2021 --gpu_id=0 --model_dim=200 --epoch=10 --base_model_dir=$out_dir/OmniAnomaly/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_50epoch_60ws_0.001lr_200model_dim --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=0.01
    # Online Detection
    python code/OmniAnomaly/get_eval_result.py --dataset_type=$dataset_type --train_type='finetune_all' --epochs=10 --training_period=1 --seed=2021 --lr=5e-4 --valid_epoch=1 --gpu_id=0 --model_dim=200 --base_model_dir=$out_dir/OmniAnomaly/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_50epoch_60ws_0.001lr_200model_dim --train_num=$train_num --out_dir=$out_dir  --min_std=0.01 

    python code/OmniAnomaly/get_eval_result.py  --dataset_type=$dataset_type --train_type='freeze_rnn_init_dense_std_2step' --training_period=1 --seed=2021 --gpu_id=0 --model_dim=200 --epoch=10 --base_model_dir=$out_dir/OmniAnomaly/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_2021_1daytrain_50epoch_60ws_0.001lr_200model_dim --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=0.01
elif test $model_name = 'SDFVAE'
then
    echo "run SDFVAE..."
    # Offline Training
    python code/SDFVAE_pytorch/example_offline.py --lr=1e-3 --dataset_type=$dataset_type --train_type='offline' --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=test --epochs=100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir --min_std=-3

    python code/SDFVAE_pytorch/example_use_center.py --lr=1e-3 --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/SDFVAE/$dataset_type/offline_${train_num}nodes_1iwi_-3.0clip_2021_1daytrain_model50_s10_d10_T5_lr0.001_weight10_epoch100 --epochs=100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=-3
    # Transfer Learning
    python code/SDFVAE_pytorch/example_finetune.py --lr=5e-4 --dataset_type=$dataset_type --train_type='finetune_all' --epochs=50 --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/SDFVAE/$dataset_type/offline_${train_num}nodes_1iwi_-3.0clip_2021_1daytrain_model50_s10_d10_T5_lr0.001_weight10_epoch100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=-3
    python code/SDFVAE_pytorch/example_finetune2step.py --lr=5e-4 --dataset_type=$dataset_type --train_type='freeze_rnn_cnn_init_decoder_linear_2step' --epochs=50 --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/SDFVAE/$dataset_type/offline_${train_num}nodes_1iwi_-3.0clip_2021_1daytrain_model50_s10_d10_T5_lr0.001_weight10_epoch100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=-3
    # Online Detection
    python code/SDFVAE_pytorch/get_eval_result.py --lr=5e-4 --dataset_type=$dataset_type --train_type='finetune_all' --epochs=50 --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/SDFVAE/$dataset_type/offline_${train_num}nodes_1iwi_-3.0clip_2021_1daytrain_model50_s10_d10_T5_lr0.001_weight10_epoch100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=-3
    python code/SDFVAE_pytorch/get_eval_result.py --lr=5e-4 --dataset_type=$dataset_type --train_type='freeze_rnn_cnn_init_decoder_linear_2step' --epochs=50 --training_period=1 --seed=2021 --gpu_id=0  --base_model_dir=$out_dir/SDFVAE/$dataset_type/offline_${train_num}nodes_1iwi_-3.0clip_2021_1daytrain_model50_s10_d10_T5_lr0.001_weight10_epoch100 --T=5 --s_dim=10 --d_dim=10 --model_dim=50 --train_num=$train_num  --index_weight_index=1 --out_dir=$out_dir  --min_std=-3

elif test $model_name = 'TranAD'
then
    echo "run TranAD..."
    # Offline Training
    python code/TranAD/example_offline.py --dataset_type=$dataset_type --epochs=100 --train_type='offline' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-4 --batch_size=256 --index_weight_index=1  --train_num=$train_num --out_dir=$out_dir --tran_dim=8 --epsilon=0.98 --valid_epoch=1

    python code/TranAD/example_use_center.py --dataset_type=$dataset_type --epochs=100 --train_type='use_center' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir="$out_dir/TranAD/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.98eps" --lr=5e-4 --batch_size=256 --train_num=$train_num --out_dir=$out_dir --tran_dim=8
    # Transfer Learning
    python code/TranAD/example_finetune.py --dataset_type=$dataset_type --epochs=10 --train_type='finetune_all' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir="$out_dir/TranAD/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.98eps" --lr=1e-4 --batch_size=256   --train_num=$train_num --out_dir=$out_dir --tran_dim=8 --valid_epoch=1

    python code/TranAD/example_finetune2step.py --dataset_type=$dataset_type --epochs=20 --train_type='freeze_att_init_last_2step' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir="$out_dir/TranAD/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.98eps" --lr=1e-4 --batch_size=256   --train_num=$train_num --out_dir=$out_dir --tran_dim=8 --valid_epoch=1
    # Online Detection
    python code/TranAD/get_eval_result.py --dataset_type=$dataset_type --epochs=10 --train_type='finetune_all' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir="$out_dir/TranAD/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.98eps" --lr=1e-4 --batch_size=256   --train_num=$train_num --out_dir=$out_dir --tran_dim=8 --valid_epoch=1

    python code/TranAD/get_eval_result.py --dataset_type=$dataset_type --epochs=20 --train_type='freeze_att_init_last_2step' --training_period=1 --seed=2021 --gpu_id=0 --base_model_dir="$out_dir/TranAD/$dataset_type/offline_${train_num}nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.98eps" --lr=1e-4 --batch_size=256   --train_num=$train_num --out_dir=$out_dir --tran_dim=8 --valid_epoch=1
elif test $model_name = 'USAD'
then
    echo "run USAD..."
    # Offline Training
    python code/USAD/example_offline.py --dataset_type=$dataset_type --train_type='offline' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=test --gpu_id=0 --epochs=100 --valid_epoch=1 --window_size=60  --z_dim=202 --valid_epoch=10  --train_num=$train_num

    python code/USAD/example_use_center.py --dataset_type=$dataset_type --train_type='use_center' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=$out_dir/USAD/$dataset_type/offline_2020_202z_${train_num}nodes_1iwi_1daytrain_0.001lr_100epoch_60ws --gpu_id=0 --epochs=100 --window_size=60  --z_dim=202 --train_num=$train_num
    # Transfer Learning
    python code/USAD/example_finetune.py --dataset_type=$dataset_type  --train_type='finetune_all' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=$out_dir/USAD/$dataset_type/offline_2020_202z_${train_num}nodes_1iwi_1daytrain_0.001lr_100epoch_60ws --gpu_id=0 --epochs=5 --valid_epoch=1 --lr=1e-4 --window_size=60 --z_dim=202 --train_num=$train_num

    python code/USAD/example_finetune2step.py --dataset_type=$dataset_type  --train_type='freeze_12_init_3_2step' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=$out_dir/USAD/$dataset_type/offline_2020_202z_${train_num}nodes_1iwi_1daytrain_0.001lr_100epoch_60ws --epochs=50 --lr=1e-3 --gpu_id=0 --valid_epoch=1 --window_size=60 --z_dim=202 --train_num=$train_num
    # Online Detection
    python code/USAD/get_eval_result.py --dataset_type=$dataset_type  --train_type='finetune_all' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=$out_dir/USAD/$dataset_type/offline_2020_202z_${train_num}nodes_1iwi_1daytrain_0.001lr_100epoch_60ws --gpu_id=0 --epochs=5 --valid_epoch=1 --lr=1e-4 --window_size=60 --z_dim=202 --train_num=$train_num

    python code/USAD/get_eval_result.py --dataset_type=$dataset_type  --train_type='freeze_12_init_3_2step' --training_period=1 --seed=2020 --out_dir=$out_dir --base_model_dir=$out_dir/USAD/$dataset_type/offline_2020_202z_${train_num}nodes_1iwi_1daytrain_0.001lr_100epoch_60ws --epochs=50 --lr=1e-3 --gpu_id=0 --valid_epoch=1 --window_size=60 --z_dim=202 --train_num=$train_num
else
    echo "no this model"
fi