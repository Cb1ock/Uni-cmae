#!/bin/bash

# run cav-mae pretraining, use smaller lr and batch size, fits smaller GPUs (4*12GB GPUs)

export TORCH_HOME=../../pretrained_models

model=uni-cmae
masking_ratioa_a=0.5
masking_ratio_v=0.9
mask_mode=unstructured # or time, or freq, or tf
contrast_loss_weight=0.01
mae_loss_weight=1.0
tr_pos=True
norm_pix_loss=True
pred_t_dim=16
decoder_depth=4
bidirect_contrast=True

cur_dir=$(pwd)
# wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O IN-initial.pth
# pretrain_path=${cur_dir}/IN-initial.pth
pretrain_path=${cur_dir}/ori_mae_12_for_pretrain.pth

lr=1e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081 # TODO: change to audioset mean and std
dataset_std=4.4849 # TODO: change to audioset mean and std
target_length=1024
noise=True
mixup=0.0
batch_size=198
lr_scheduler=cosine
warmup_epochs=5

dataset=voxceleb2
tr_data=/home/hao/Project/uni-cmae/egs/voxceleb2/train_data.json
te_data=/home/hao/Project/uni-cmae/egs/voxceleb2/test_data.json
label_csv=/home/hao/Project/uni-cmae/egs/voxceleb2/class_labels_indices.csv

timestamp=$(date +%Y%m%d%H%M%S)
exp_dir=./exp/testmae02-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${masking_ratio_v}-${masking_ratioa_a}--${mask_mode}-bidirect_contrast-${bidirect_contrast}-croped-cacvnorm
mkdir -p $exp_dir

#cont_model=/home/hao/Project/uni-cmae/egs/voxceleb2/exp/testmae02-audioset-uni-cmae-balNone-lr5e-5-epoch25-bs56-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/audio_model.8.pth

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -W ignore ../../src/run_unicmae_pretrain.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label_csv ${label_csv} --n_class 527 --pred_t_dim ${pred_t_dim} --decoder_depth ${decoder_depth} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True --warmup_epochs ${warmup_epochs} \
--mixup ${mixup} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} --lr_scheduler ${lr_scheduler} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio_a ${masking_ratioa_a} --masking_ratio_v ${masking_ratio_v} --mask_mode ${mask_mode} >> $exp_dir/train.log 2>&1