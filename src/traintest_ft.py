# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import csv
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler

def train(audio_model, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_metric = 0, -np.inf

    n_print_steps = max(len(train_loader) // args.batch_size // 10, 100)
    
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_metric, time.time() - start_time])
        with open(f"{exp_dir}/progress.pkl", "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # 定义 MLP 层的名称列表（新初始化的层）
    mlp_list = [
        'mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
        'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
        'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
        'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
        'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias'
    ]
    mlp_params = [param for name, param in audio_model.module.named_parameters() if name in mlp_list]
    base_params = [param for name, param in audio_model.module.named_parameters() if name not in mlp_list]

    # 如果需要，冻结预训练的参数
    if args.freeze_base:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized MLP layer uses {:.3f} x larger lr'.format(args.head_lr))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr}
        ], weight_decay=5e-7, betas=(0.95, 0.999))

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr}
        ], weight_decay=5e-7, momentum=0.9)

    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr},
        {'params': mlp_params, 'lr': args.lr * args.head_lr}
        ], weight_decay=5e-7, betas=(0.95, 0.999))


    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [base_lr, mlp_lr]
    print('optimizer, base lr, mlp lr : ', args.optimizer, base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # 学习率调度器
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Using adaptive learning rate scheduler.')
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)), gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at epoch {:d} with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
        print('Using cosine annealing learning rate scheduler.')

    main_metrics = args.metrics

    # 加载类别权重
    if args.balance == 'bal':
        print(f'loading class weights from {args.weight_file}')
        class_weights = np.load(args.weight_file)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights_tensor = None

    # 选择损失函数
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor) if args.balance == 'bal' else nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor) if args.balance == 'bal' else nn.CrossEntropyLoss()

    args.loss_fn = loss_fn


    print('Now training with dataset: {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(
        str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)
    ))

    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])

    while epoch < args.n_epochs :
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, v_input, labels) in enumerate(train_loader):

            B = a_input.size(0)
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)
                loss = loss_fn(audio_output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / B)
            per_sample_dnn_time.update((time.time() - dnn_start_time) / B)

            print_step = global_step % n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (n_print_steps // 10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                      f'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      f'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      f'Train Loss {loss_meter.val:.4f}\t', flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')

        # 获取验证结果和损失
        metrics_result, valid_loss = validate(audio_model, val_loader, args)

        # 提取指标
        accuracy = metrics_result['accuracy']
        weighted_recall = metrics_result['weighted']['recall']
        macro_recall = metrics_result['macro']['recall']

        print(f"Accuracy: {accuracy:.6f}")
        print(f"Weighted Recall (WAR): {weighted_recall:.6f}")
        print(f"Macro Recall (UAR): {macro_recall:.6f}")
        print(f"Training loss: {loss_meter.avg:.6f}")
        print(f"Validation loss: {valid_loss:.6f}")

        result[epoch - 1, :] = [accuracy, weighted_recall, macro_recall, optimizer.param_groups[0]['lr']]
        np.savetxt(f"{exp_dir}/result.csv", result, delimiter=',')
        print('validation finished')

        # 根据主指标确定是否保存模型
        current_metric = None
        if main_metrics == 'war':
            current_metric = weighted_recall
        elif main_metrics == 'acc':
            current_metric = accuracy
        elif main_metrics == 'uar':
            current_metric = macro_recall

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            # 保存最佳模型
            torch.save(audio_model.state_dict(), f"{exp_dir}/models/best_audio_model.pth")
            torch.save(optimizer.state_dict(), f"{exp_dir}/models/best_optim_state.pth")

        if args.save_model:
            torch.save(audio_model.state_dict(), f"{exp_dir}/models/audio_model.{epoch}.pth")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_metric)
        else:
            scheduler.step()

        print(f'Epoch-{epoch} lr: {optimizer.param_groups[0]["lr"]}')

        # # 保存指标结果
        # with open(f"{exp_dir}/stats_{epoch}.pickle", 'wb') as handle:
        #     pickle.dump(metrics_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        # 重置计数器
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def calculate_war_uar(stats, target_labels, predictions):
    # 计算加权平均的 precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_labels, predictions, average='weighted')
    
    # 计算宏平均的 precision, recall, f1
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_labels, predictions, average='macro')
    
    # 计算准确率
    accuracy = metrics.accuracy_score(target_labels, predictions)
    
    # 将结果返回
    return {
        'weighted': {'precision': precision, 'recall': recall, 'f1': f1},
        'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'accuracy': accuracy
    }

def validate(audio_model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats, target_labels, predictions = calculate_stats(audio_output, target)
        metrics_result = calculate_war_uar(stats, target_labels, predictions)

    if output_pred == False:
        return metrics_result, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return metrics_result, audio_output, target

def test(audio_model, test_loader, args, num_tests=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()
    all_metrics = []
    all_losses = []
    all_stats = []

    for test_idx in range(num_tests):
        batch_time = AverageMeter()
        end = time.time()
        A_predictions, A_targets, A_loss = [], [], []
        with torch.no_grad():
            for i, (a_input, v_input, labels) in enumerate(test_loader):
                a_input = a_input.to(device)
                v_input = v_input.to(device)

                with autocast():
                    audio_output = audio_model(a_input, v_input, args.ftmode)

                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                labels = labels.to(device)
                loss = args.loss_fn(audio_output, labels)
                A_loss.append(loss.to('cpu').detach())

                batch_time.update(time.time() - end)
                end = time.time()

            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            loss = np.mean(A_loss)

            stats, target_labels, predictions = calculate_stats(audio_output, target)
            metrics_result = calculate_war_uar(stats, target_labels, predictions)
            all_metrics.append(metrics_result)
            all_losses.append(loss)
            all_stats.append(stats)

        print(f"Test {test_idx + 1} Accuracy: {metrics_result['accuracy']:.6f}")
        print(f"Test {test_idx + 1} Weighted Recall (WAR): {metrics_result['weighted']['recall']:.6f}")
        print(f"Test {test_idx + 1} Macro Recall (UAR): {metrics_result['macro']['recall']:.6f}")
        # for k, v in stats.items():
        #     if k != 'overall':
        #         print(f"Class {k} - Precision: {v['precision']:.6f}, Recall: {v['recall']:.6f}, F1: {v['f1']:.6f}, Sample Count: {v['sample_count']}")
        print(f"Test {test_idx + 1} Loss: {loss:.6f}")
    
    # 计算并输出平均指标
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_war = np.mean([m['weighted']['recall'] for m in all_metrics])
    avg_uar = np.mean([m['macro']['recall'] for m in all_metrics])
    avg_loss = np.mean(all_losses)
    avg_stats = {k: {m: np.mean([s[k][m] for s in all_stats]) for m in ['precision', 'recall', 'f1']} for k in all_stats[0].keys() if k != 'overall'}
    print(f"Average Accuracy: {avg_accuracy:.6f}")
    print(f"Average Weighted Recall (WAR): {avg_war:.6f}")
    print(f"Average Macro Recall (UAR): {avg_uar:.6f}")

    label_to_class = {}
    with open(args.label_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            index, mid, display_name = row
            label_to_class[int(index)] = display_name

    for k, v in avg_stats.items():
        print(f"Average Class {label_to_class[k]} - Precision: {v['precision']:.6f}, Recall: {v['recall']:.6f}, F1: {v['f1']:.6f}")
    print(f"Average Loss: {avg_loss:.6f}")
