# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler, Subset
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_ft import train, validate, test
from collections import Counter

# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-test", type=str, default=None, help="test data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "MAFW", "DFEW","CREMA-D","fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument('--num_tests', default=1, type=int, metavar='N', help='number of tests to run')

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optimizer", type=str, default="adam", help="training optimizer", choices=["adam","sgd","sgd+momentum","adamw"])
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc", 'uar', 'war','waf'])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
parser.add_argument('--drop_path', type=float, default=0.0, help='Stochastic depth rate')

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--balance", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

parser.add_argument("--dataset_type", type=str, default='frame', help="the dataset type used", choices=["frame", "video"])
parser.add_argument("--im_res", type=int, default=160, help="the image resolution")
parser.add_argument("--lr_scheduler", type=str, default='cosine', help="the learning rate scheduler", choices=["cosine", "step", "plateau"])
parser.add_argument("--debug", help="enable debug mode", type=ast.literal_eval, default=False)
args = parser.parse_args()

# all exp in this work is based on 224 * 224 image
im_res = args.im_res
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':'finetune', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'val', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
test_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                   'mode':'test', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
video_conf = {
    'dataset_type' : args.dataset_type, 
    'aa_type': "rand-m7-n4-mstd0.5-inc1",
    'pretrain_rand_flip': True,
    'pretrain_rand_erase_prob': 0.25,
    'pretrain_rand_erase_mode': "pixel",
    'pretrain_rand_erase_count': 1,
    'pretrain_rand_erase_split': False,
    'jitter_aspect_relative': [0.75, 1.3333],
    'jitter_scales_relative': [0.5, 1.0],
    'repeat_aug': 1,
    'num_retries': 10,
    'train_jitter_scales': (224, 224),
    'train_crop_size': im_res,
    'train_random_horizontal_flip': True,
    'test_num_ensemble_views': 10,
    'test_num_spatial_crops': 3,
    'test_crop_size': im_res, # TODO：可以改，主要就是位置编码那地方和patch_embedding那地方
    'sampling_rate': 4,
    'num_frames': 16,
    'target_fps': 30,
    'mean': (0.45, 0.45, 0.45),
    'std': (0.225, 0.225, 0.225),
    'enable_multi_thread_decode': False,
    'inverse_uniform_sampling': False,
    'use_offset_sampling': True
}

def get_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last, debug):
    if debug:
        dataset = Subset(dataset, range(100))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
def get_labels_and_weights(dataset):
    """
    获取数据集的标签和样本权重
    """
    labels = []
    for _, _, label in dataset:  # 假设dataset[i]返回(input, label)
        labels.append(label)
    
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return labels, weights

if args.balance == 'bal':
    print('balanced sampler is being used')
    train_dataset = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, video_conf=video_conf)
    
    # 获取标签和权重
    labels, samples_weight = get_labels_and_weights(train_dataset)
    
    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)


    train_dataset = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, video_conf=video_conf)
    
    train_dataset = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, video_conf=video_conf)
    if args.debug:
        train_dataset = Subset(train_dataset, range(100))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_dataset = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, video_conf=video_conf)
    if args.debug:
        train_dataset = Subset(train_dataset, range(100))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_dataset = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, video_conf=video_conf)
val_loader = get_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, debug=args.debug)

if args.data_test != None:
    test_dataset = dataloader.AudiosetDataset(args.data_test, label_csv=args.label_csv, audio_conf=test_audio_conf, video_conf=video_conf)
    test_loader = get_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, debug=args.debug)

if args.model == 'uni-cmae-ft':
    print('finetune a uni model with 12 layers')
    audio_model = models.Uni_CMAEFT(label_dim=args.n_class, encoder_depth=12, drop_path=args.drop_path,img_size=160)
else:
    raise ValueError('model not supported')

if args.pretrain_path == 'None':
    warnings.warn("Note you are finetuning a model without any finetuning.")

# finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
if args.pretrain_path != 'None':
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.pretrain_path)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load cav-mae pretrained weights from ', args.pretrain_path)
    print(msg)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir, start_epoch, end_epoch):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA

# evaluate with multiple frames
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)
if args.wa == True:
    sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end)
    torch.save(sdA, args.exp_dir + "/models/audio_model_wa.pth")
    # 删除其他模型文件
    for epoch in range(args.wa_start, args.wa_end + 1):
        os.remove(args.exp_dir + '/models/audio_model.' + str(epoch) + '.pth')
else:
    # 如果没有加权平均，使用最佳检查点
    sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')

msg = audio_model.load_state_dict(sdA, strict=True)
print(msg)
audio_model.eval()

test(audio_model, test_loader, args, num_tests=args.num_tests)