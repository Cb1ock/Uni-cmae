#!/bin/bash

# run uni-cmae finetuning

export TORCH_HOME=../../pretrained_models

model=uni-cmae-ft
ftmode=audioonly # multimodal or audioonly or videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)


freeze_base=False
head_lr=50 # newly initialized ft layers uses 50 times larger than the base lr

bal=None
lr=1e-5
epoch=30
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=1
wa_end=10
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0
batch_size=60
label_smooth=0.1

dataset=MAFW
tr_data=/home/hao/Project/uni-cmae/egs/MAFW/train_data.json
te_data=/home/hao/Project/uni-cmae/egs/MAFW/test_data.json
label_csv=/home/hao/Project/uni-cmae/egs/MAFW/class_labels_indices.csv

pretrain_path="/home/hao/Project/uni-cmae/egs/voxceleb2/exp/testmae02-audioset-uni-cmae-balNone-lr5e-5-epoch6-bs40-normTrue-c0.01-p1.0-tpFalse-mr-0.9-0.75/models/best_audio_model.pth"

exp_dir=./exp/testmae01-full-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-war
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -W ignore ../../src/run_unicmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 11 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --im_res 160 --dataset_type frame \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics war --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32 >> $exp_dir/train.log 2>&1