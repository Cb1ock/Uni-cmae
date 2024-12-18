#!/bin/bash

# run uni-cmae finetuning

export TORCH_HOME=../../pretrained_models

model=uni_cmae_ablation # or uni_cmae_ablation
ftmode=multimodal # multimodal or audioonly or videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)

freeze_base=False
head_lr=50 # newly initialized ft layers uses 50 times larger than the base lr

balance=no
lr=1e-5
lr_scheduler=cosine
warmup_epochs=2.5

epoch=25
wa=True
wa_start=10
wa_end=25
tr_pos=True

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
optimizer=adam
noise=True
freqm=48
timem=192
mixup=0.2
batch_size=12
label_smooth=0.3
drop_path=0
num_tests=10

dataset=MAFW
label_csv=/home/hao/Project/uni-cmae/egs/MAFW/class_labels_indices.csv

pretrain_path="/home/hao/Project/uni-cmae/pretrained_model/UniCmae_2LayerFusion_bs50_PixnormTrue_tpTrue_c0.01_0.9_0.5_1e-4_16frame.pth"

for fold in {1..5}
do
    tr_data=/home/hao/Project/uni-cmae/egs/MAFW/cross_validation/fold_${fold}_train.json
    val_data=/home/hao/Project/uni-cmae/egs/MAFW/cross_validation/fold_${fold}_val.json
    data_test=/home/hao/Project/uni-cmae/egs/MAFW/cross_validation/fold_${fold}_val.json
    weight_file=/home/hao/Project/uni-cmae/egs/MAFW/cross_validation/fold_${fold}_train_weights.npy
    exp_dir=./exp/2LayerFusion-${model}-${balance}-${lr}-${optimizer}-bs${batch_size}--${ftmode}-mx${mixup}-dp${drop_path}-fz${freeze_base}-h${head_lr}-war/fold${fold}
    mkdir -p $exp_dir

    CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6, python -W ignore ../../src/run_unicmae_ft.py --model ${model} --dataset ${dataset} \
    --data_train ${tr_data} --data-val ${val_data} --data-test ${data_test} --exp-dir $exp_dir \
    --label_csv ${label_csv} --n_class 11 --num_tests ${num_tests} \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True --tr_pos ${tr_pos} \
    --freqm $freqm --timem $timem --mixup ${mixup} --balance ${balance} --im_res 160 --dataset_type frame --drop_path ${drop_path} \
    --label_smooth ${label_smooth} --weight_file ${weight_file} \
    --optimizer ${optimizer} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss CE --metrics war --warmup True \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_scheduler ${lr_scheduler} \
    --pretrain_path ${pretrain_path} --ftmode ${ftmode} --debug False \
    --freeze_base ${freeze_base} --head_lr ${head_lr} \
    --num-workers 32 > $exp_dir/train.log 2>&1
done
