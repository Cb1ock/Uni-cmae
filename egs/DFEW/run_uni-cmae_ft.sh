#!/bin/bash

# run uni-cmae finetuning

export TORCH_HOME=../../pretrained_models

model=uni-cmae-ft
ftmode=multimodal # multimodal or audioonly or videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)

freeze_base=False
head_lr=50 # newly initialized ft layers uses 50 times larger than the base lr

balance=no
lr=5e-5
epoch=10
lr_scheduler=cosine
warmup_epochs=2.5
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=1
wa_end=10

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
optimizer=sgd
noise=True
freqm=48
timem=192
mixup=0
batch_size=60
label_smooth=0.3
drop_path=0.1

num_tests=10

dataset=DFEW
label_csv=/data/heyichao/uni-cmae1/egs/DFEW/class_labels_indices.csv

pretrain_path="/data/hao/model/best_audio_model.pth"

# for fold in {1..5}
# do
#     tr_data=/data/heyichao/uni-cmae2/egs/DFEW/train_set_${fold}.json
#     val_data=/data/heyichao/uni-cmae2/egs/DFEW/test_set_${fold}.json
#     data_test=/data/heyichao/uni-cmae2/egs/DFEW/test_set_${fold}.json
#     exp_dir=./exp/full-${model}-${balance}-${lr}-${mixup}-${drop_path}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}--${ftmode}-fz${freeze_base}-h${head_lr}-war/fold${fold}
#     mkdir -p $exp_dir

#     CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -W ignore ../../src/run_unicmae_ft.py --model ${model} --dataset ${dataset} \
#     --data-train ${tr_data} --data-val ${val_data} --data-test ${data_test} --exp-dir $exp_dir \
#     --label_csv ${label_csv} --n_class 7 --num_tests ${num_tests} \
#     --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
#     --freqm $freqm --timem $timem --mixup ${mixup} --balance ${balance} --im_res 160 --dataset_type frame \
#     --label_smooth ${label_smooth} \
#     --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} --optimizer ${optimizer} \
#     --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
#     --loss CE --metrics war --warmup True \
#     --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_scheduler ${lr_scheduler} \
#     --pretrain_path ${pretrain_path} --ftmode ${ftmode} --debug False \
#     --freeze_base ${freeze_base} --head_lr ${head_lr} \
#     --num-workers 32 >> $exp_dir/train.log 2>&1
# done

fold=1

tr_data=/data/heyichao/uni-cmae2/egs/DFEW/train_set_${fold}.json
val_data=/data/heyichao/uni-cmae2/egs/DFEW/test_set_${fold}.json
data_test=/data/heyichao/uni-cmae2/egs/DFEW/test_set_${fold}.json
exp_dir=./exp/full-${model}-${balance}-${lr}-${mixup}-${drop_path}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}--${ftmode}-fz${freeze_base}-h${head_lr}-war/fold${fold}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -W ignore ../../src/run_unicmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-test ${data_test} --exp-dir $exp_dir \
--label_csv ${label_csv} --n_class 7 --num_tests ${num_tests} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --balance ${balance} --im_res 160 --dataset_type frame \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} --optimizer ${optimizer} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics war --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_scheduler ${lr_scheduler} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} --debug False \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32 >> $exp_dir/train.log 2>&1
