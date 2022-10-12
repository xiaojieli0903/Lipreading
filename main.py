#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
""" TCN for lipreading"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lipreading.dataloaders import (get_data_loaders,
                                    get_preprocessing_pipelines)
from lipreading.mixup import mixup_criterion, mixup_data
from lipreading.model import Lipreading
from lipreading.optim_utils import CosineScheduler, get_optimizer
from lipreading.utils import (AverageMeter, CheckpointSaver, calculateNorm2,
                              get_logger, get_save_folder, load_json,
                              load_model, save2npz, showLR,
                              update_logger_batch)


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes',
                        type=int,
                        default=500,
                        help='Number of classes')
    parser.add_argument('--modality',
                        default='video',
                        choices=['video', 'audio'],
                        help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir',
                        default='./datasets/LRW_h96w96_mouth_crop_gray',
                        help='Loaded data directory')
    parser.add_argument('--label-path',
                        type=str,
                        default='./labels/500WordsSortedList.txt',
                        help='Path to txt file with labels')
    parser.add_argument('--annonation-direc',
                        default=None,
                        help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type',
                        type=str,
                        default='resnet',
                        choices=['resnet', 'shufflenet'],
                        help='Architecture used for backbone')
    parser.add_argument('--relu-type',
                        type=str,
                        default='relu',
                        choices=['relu', 'prelu'],
                        help='what relu to use')
    parser.add_argument('--width-mult',
                        type=float,
                        default=1.0,
                        help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size',
                        type=int,
                        nargs="+",
                        help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers',
                        type=int,
                        default=4,
                        help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout',
                        type=float,
                        default=0.2,
                        help='Dropout value for the TCN module')
    parser.add_argument(
        '--tcn-dwpw',
        default=False,
        action='store_true',
        help=
        'If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult',
                        type=int,
                        default=1,
                        help='TCN width multiplier')
    # -- DenseTCN config
    parser.add_argument('--densetcn-block-config',
                        type=int,
                        nargs="+",
                        help='number of denselayer for each denseTCN block')
    parser.add_argument('--densetcn-kernel-size-set',
                        type=int,
                        nargs="+",
                        help='kernel size set for each denseTCN block')
    parser.add_argument('--densetcn-dilation-size-set',
                        type=int,
                        nargs="+",
                        help='dilation size set for each denseTCN block')
    parser.add_argument('--densetcn-growth-rate-set',
                        type=int,
                        nargs="+",
                        help='growth rate for DenseTCN')
    parser.add_argument('--densetcn-dropout',
                        default=0.2,
                        type=float,
                        help='Dropout value for DenseTCN')
    parser.add_argument('--densetcn-reduced-size',
                        default=256,
                        type=int,
                        help='the feature dim for the output of reduce layer')
    parser.add_argument('--densetcn-se',
                        default=False,
                        action='store_true',
                        help='If True, enable SE in DenseTCN')
    parser.add_argument('--densetcn-condense',
                        default=False,
                        action='store_true',
                        help='If True, enable condenseTCN')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Mini-batch size')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--lr',
                        default=3e-4,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('--init-epoch',
                        default=0,
                        type=int,
                        help='epoch to start at')
    parser.add_argument('--epochs',
                        default=80,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='training mode')
    # -- mixup
    parser.add_argument('--alpha',
                        default=0.4,
                        type=float,
                        help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path',
                        type=str,
                        default=None,
                        help='Pretrained model pathname')
    parser.add_argument(
        '--allow-size-mismatch',
        default=False,
        action='store_true',
        help=
        'If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes'
    )
    # -- feature extractor
    parser.add_argument('--extract-feats',
                        default=False,
                        action='store_true',
                        help='Feature extractor')
    parser.add_argument(
        '--mouth-patch-path',
        type=str,
        default=None,
        help='Path to the mouth ROIs, assuming the file is saved as numpy.array'
    )
    parser.add_argument('--mouth-embedding-out-path',
                        type=str,
                        default=None,
                        help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path',
                        type=str,
                        default=None,
                        help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval',
                        default=50,
                        type=int,
                        help='display interval')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='number of data loading workers')
    # paths
    parser.add_argument(
        '--logging-dir',
        type=str,
        default='./train_logs',
        help='path to the directory in which to save the log file')
    # use boundaries
    parser.add_argument('--use-boundary',
                        default=False,
                        action='store_true',
                        help='include hard border at the testing stage.')
    # exp name
    parser.add_argument('--exp-name',
                        type=str,
                        default='',
                        help='the name of the exp.')
    # predict loss weight
    parser.add_argument('--predict-loss-weight',
                        type=float,
                        default=1.0,
                        help='the weight of the prediction loss.')

    # cls loss weight
    parser.add_argument('--cls-loss-weight',
                        type=float,
                        default=1.0,
                        help='the weight of the cls loss.')
    # loss average dim
    parser.add_argument('--loss-average-dim',
                        type=int,
                        default=-1,
                        help='the average dim of the L2 loss.')
    # detach target
    parser.add_argument('--detach-target',
                        default=False,
                        action='store_true',
                        help='detach the target when calculate loss.')
    args = parser.parse_args()
    return args


args = load_args()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True


def extract_feats(model, path_list):
    """
    :rtype: FloatTensor
    """
    model.eval()
    lines = open(path_list, 'r').readlines()
    idx = 0
    for line in lines:
        if idx % 1000 == 0:
            print(f"processing idx {idx}")
        single_path = line.strip()
        preprocessing_func = get_preprocessing_pipelines(args.modality)['test']
        data = preprocessing_func(np.load(single_path)['data'])  # data: TxHxW
        idx += 1
        data_name = single_path.split('/')[-1]
        out_path = os.path.join(args.mouth_embedding_out_path, data_name)
        save2npz(out_path,
                 model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]]).cpu().detach().numpy())


def l2_loss(pred, target, average_dim=-1):
    """L2 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        average_dim (int): The average dim of loss.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    if average_dim == -1:
        loss = torch.sum(torch.pow(pred - target, 2)) / target.numel()
    else:
        loss = torch.sum(torch.pow(pred - target, 2)) / pred.shape[average_dim]
    return loss


def evaluate(model, dset_loader, criterion):

    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader)):
            if args.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None
            if model.predict_future >= 0:
                logits, feature_predict, feature_target = model(
                    input.unsqueeze(1).cuda(),
                    lengths=lengths,
                    boundaries=boundaries)
            else:
                logits = model(input.unsqueeze(1).cuda(),
                               lengths=lengths,
                               boundaries=boundaries)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(
                labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)
    logits = preds = input = labels = None

    print(
        f"{len(dset_loader.dataset)} in total\tCR: {running_corrects/len(dset_loader.dataset)}"
    )
    return running_corrects / len(dset_loader.dataset), running_loss / len(
        dset_loader.dataset)


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info(f"Epoch {epoch}/{args.epochs - 1}")
    logger.info(f"Current learning rate: {lr}")

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.
    loss_dict = {}
    loss_weight = {}

    end = time.time()
    for batch_idx, data in enumerate(dset_loader):
        if args.use_boundary:
            input, lengths, labels, boundaries = data
            boundaries = boundaries.cuda()
        else:
            input, lengths, labels = data
            boundaries = None

        lr = showLR(optimizer)
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()
        loss = torch.zeros(1).float().cuda()
        if model.predict_future >= 0:
            logits, feature_predict, feature_target = model(
                input.unsqueeze(1).cuda(),
                lengths=lengths,
                boundaries=boundaries,
                targets=labels_a)
            if args.detach_target:
                loss_predict = l2_loss(feature_predict, feature_target.detach(), args.loss_average_dim)
            else:
                loss_predict = l2_loss(feature_predict, feature_target, args.loss_average_dim)
            loss_dict['loss_L2'] = loss_predict
            loss_weight['loss_L2'] = args.predict_loss_weight
            loss += args.predict_loss_weight * loss_predict
        else:
            logits = model(input.unsqueeze(1).cuda(),
                           lengths=lengths,
                           boundaries=boundaries,
                           targets=labels_a)

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss_KL = loss_func(criterion, logits)
        loss_dict['loss_KL'] = loss_KL
        loss_weight['loss_KL'] = args.cls_loss_weight
        loss += args.cls_loss_weight * loss_KL

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item() * input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(
            predicted)).sum().item() + (1 - lam) * predicted.eq(
                labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx
                                              == len(dset_loader) - 1):
            update_logger_batch(
                args, logger, dset_loader, batch_idx, running_loss, loss_dict,
                loss_weight, running_corrects, running_all, batch_time,
                data_time, lr,
                torch.cuda.max_memory_allocated() / 1024 / 1024)

    return model


def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json(args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)
    args.linear_config = args_loaded.get("linear_config",
                                         {'linear_type': 'Linear'})
    args.predict_future = args_loaded.get("predict_future", -1)
    args.frontend_type = args_loaded.get("frontend_type", '3D')
    args.use_memory = args_loaded.get("use_memory", False)
    args.membanks_size = args_loaded.get("membanks_size", 1024)
    args.predict_residual = args_loaded.get("predict_residual", False)
    args.predict_type = args_loaded.get("predict_type", 0)
    args.block_size = args_loaded.get("block_size", 4)
    args.memory_options = args_loaded.get("memory_options", {'radius': 16,
                                                             'slot': 112,
                                                             'head': 8})

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = {
            'num_layers': args_loaded['tcn_num_layers'],
            'kernel_size': args_loaded['tcn_kernel_size'],
            'dropout': args_loaded['tcn_dropout'],
            'dwpw': args_loaded['tcn_dwpw'],
            'width_mult': args_loaded['tcn_width_mult'],
        }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {
            'block_config': args_loaded['densetcn_block_config'],
            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
            'reduced_size': args_loaded['densetcn_reduced_size'],
            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
            'squeeze_excitation': args_loaded['densetcn_se'],
            'dropout': args_loaded['densetcn_dropout'],
        }
    else:
        densetcn_options = {}

    model = Lipreading(modality=args.modality,
                       num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       densetcn_options=densetcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       use_boundary=args.use_boundary,
                       extract_feats=args.extract_feats,
                       linear_config=args.linear_config,
                       predict_future=args.predict_future,
                       frontend_type=args.frontend_type,
                       use_memory=args.use_memory,
                       membanks_size=args.membanks_size,
                       predict_residual=args.predict_residual,
                       predict_type=args.predict_type,
                       block_size=args.block_size,
                       memory_options=args.memory_options
                       ).cuda()
    calculateNorm2(model)
    return model


def main():

    # -- logging
    save_path = get_save_folder(args)
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    model = get_model_from_json()
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)

    if args.model_path:
        assert args.model_path.endswith('.pth') and os.path.isfile(args.model_path), \
            f"'.pth' model path does not exist. Path input: {args.model_path}"
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(
                args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info(
                f'Model and states have been successfully loaded from {args.model_path}'
            )
        # init from trained model
        else:
            model = load_model(args.model_path,
                               model,
                               allow_size_mismatch=args.allow_size_mismatch)
            logger.info(
                f'Model has been successfully loaded from {args.model_path}')
        # feature extraction
        if args.mouth_patch_path:
            extract_feats(model, args.mouth_patch_path)
            return
        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'],
                                                   criterion)
            logger.info(
                f"Test-time performance on partition {'test'}: Loss: {loss_avg_test:.4f}\tAcc:{acc_avg_test:.4f}"
            )
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch - 1)

    epoch = args.init_epoch
    logger.info(model)
    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch,
                      optimizer, logger)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'],
                                             criterion)
        logger.info(
            f"{'val'} Epoch:\t{epoch:2}\tLoss val: {loss_avg_val:.4f}\tAcc val:{acc_avg_val:.4f}, LR: {showLR(optimizer)}"
        )
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'],
                                           criterion)
    logger.info(
        f"Test time performance of best epoch: {acc_avg_test} (loss: {loss_avg_test})"
    )


if __name__ == '__main__':
    main()
