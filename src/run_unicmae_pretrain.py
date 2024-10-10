# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
from traintest_cavmae import train

# pretrain cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label_csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "voxceleb2", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
parser.add_argument("--dataset_mean", type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments, only for preliminary experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_scheduler", help='which lr scheduler to use', type=str, default='step', choices=['plateau', 'step', 'cosine'])
parser.add_argument("--warmup_epochs", type=int, default=0, help="how many epoch to warmup")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")

parser.add_argument("--pred_t_dim", type=int, default=8, help="the number of frames to predict in the future")
parser.add_argument("--encoder_depth", type=int, default=12, help="the depth of the encoder")
parser.add_argument('--fusion_depth', help='the depth of the fusion layer', type=int, default=2)
parser.add_argument("--decoder_depth", type=int, default=8, help="the depth of the decoder")
parser.add_argument("--bidirect_contrast", type=ast.literal_eval, default=False, help="if use bidirectional contrastive loss")

parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)
parser.add_argument("--masking_ratio_a", type=float, default=0.5, help="audio masking ratio")
parser.add_argument("--masking_ratio_v", type=float, default=0.9, help="video masking ratio")
parser.add_argument("--mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])

args = parser.parse_args()

im_res = 160
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'pretrain', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'val', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
video_conf = {
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
    'train_jitter_scales': (256, 320),
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
print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

def collate_fn(batch):
    max_fbank_length = max([sample[0].shape[0] for sample in batch])
    max_image_shape = [max([sample[1].shape[i] for sample in batch]) for i in range(4)]
    
    # 填充或裁剪数据
    collated_fbank = []
    collated_image = []
    labels = []

    for fbank, image, label in batch:
        # 填充 fbank 数据
        padded_fbank = torch.nn.functional.pad(fbank, (0, 0, 0, max_fbank_length - fbank.shape[0]))
        collated_fbank.append(padded_fbank)

        # 填充 image 数据
        padded_image = torch.nn.functional.pad(image, (0, max_image_shape[3] - image.shape[3], 0, max_image_shape[2] - image.shape[2], 0, max_image_shape[1] - image.shape[1], 0, max_image_shape[0] - image.shape[0]))
        collated_image.append(padded_image)

        # 收集标签
        labels.append(label)

    # 将列表转换为张量
    collated_fbank = torch.stack(collated_fbank)
    collated_image = torch.stack(collated_image)
    labels = torch.tensor(labels)

    return collated_fbank, collated_image, labels


train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf,video_conf=video_conf),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf,video_conf=video_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf,video_conf=video_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

if args.model == 'uni-cmae':
    print('pretrain a uni model with 12 layers')
    audio_model = models.Uni_CMAE(
                                img_size=im_res, 
                                audio_length=args.target_length, 
                                norm_pix_loss=args.norm_pix_loss, 
                                encoder_depth=args.encoder_depth, 
                                decoder_depth = args.decoder_depth, 
                                tr_pos=args.tr_pos, 
                                pred_t_dim=args.pred_t_dim,
                                bidirect_contrast=args.bidirect_contrast,
                                )
elif args.model == 'uni-cmae-ablation':
    audio_model = models.Uni_CMAE_ablation(
                                img_size=im_res, 
                                audio_length=args.target_length, 
                                norm_pix_loss=args.norm_pix_loss, 
                                encoder_depth=args.encoder_depth, 
                                fusion_depth=args.fusion_depth,
                                decoder_depth = args.decoder_depth, 
                                tr_pos=args.tr_pos, 
                                pred_t_dim=args.pred_t_dim,
                                bidirect_contrast=args.bidirect_contrast,
                                )
elif args.model == 'cav-mae':
    audio_model = models.cav_mae(
                                img_size=im_res, 
                                audio_length=args.target_length, 
                                norm_pix_loss=args.norm_pix_loss, 
                                encoder_depth=args.encoder_depth, 
                                fusion_depth=args.fusion_depth,
                                decoder_depth = args.decoder_depth, 
                                tr_pos=args.tr_pos, 
                                pred_t_dim=args.pred_t_dim,
                                bidirect_contrast=args.bidirect_contrast,
                                )
else:
    raise ValueError('model not supported')

# initialized with a pretrained checkpoint (e.g., original vision-MAE checkpoint)
if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    # 处理键名不匹配问题
    new_state_dict = {}
    for k, v in mdl_weight.items():
        if not k.startswith('module.'):
            k = 'module.' + k
        new_state_dict[k] = v
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    
    # 加载状态字典并跳过形状不匹配或不存在的参数
    model_dict = audio_model.state_dict()
    for name, param in new_state_dict.items():
        if name in model_dict:
            if model_dict[name].shape != param.shape:
                print(f"Skipping parameter {name} due to shape mismatch: {param.shape} vs {model_dict[name].shape}")
                continue
            model_dict[name].copy_(param)
        else:
            print(f"Skipping parameter {name} as it does not exist in the model")
    msg = audio_model.load_state_dict(model_dict, strict=False)
    
    print('now load mae pretrained weights from ', args.pretrain_path)
    print(msg)

if args.cont_model != None:
    print('now load pretrained weights from : ' + args.cont_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdA = torch.load(args.cont_model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sdA, strict=True)

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