# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
import os
from xml.sax.handler import all_features
import numpy as np
import torch
import datasets
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from datasets import build_dataset, build_processed_dataset
from engine import preprocess_neighbors
from engine import encoder_evaluate_swig, evaluate_swig
from engine import encoder_train_one_epoch, decoder_train_one_epoch
from models import build_encoder_model, build_decoder_model
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set Grounded Situation Recognition Transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--encoder_batch_size', default=16, type=int)
    parser.add_argument('--decoder_batch_size', default=4, type=int)
    parser.add_argument('--encoder_epochs', default=0, type=int)
    parser.add_argument('--decoder_epochs', default=0, type=int)

    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--num_enc_layers', default=6, type=int,
                        help="Number of encoding layers in GSRFormer")
    parser.add_argument('--num_dec_layers', default=5, type=int,
                        help="Number of decoding layers in GSRFormer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Loss coefficients
    parser.add_argument('--noun_loss_coef', default=2, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_conf_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=5, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--swig_path', type=str, default="SWiG")
    parser.add_argument('--dev', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    # Etc...
    parser.add_argument('--inference', default=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--encoder_start_epoch', default=0, type=int, metavar='N',
                        help='encoder start epoch')
    parser.add_argument('--decoder_start_epoch', default=0, type=int, metavar='N',
                        help='decoder start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_saved_model', default='GSRFormer/encoder_checkpoint.pth',
                        help='path where saved encoder model is')
    parser.add_argument('--decoder_saved_model', default='GSRFormer/decoder_checkpoint.pth',
                        help='path where saved decoder model is')    
    parser.add_argument('--load_saved_encoder', default=False, type=bool)
    parser.add_argument('--load_saved_decoder', default=False, type=bool)
    parser.add_argument('--preprocess', default=False, type=bool)
    parser.add_argument('--images_per_segment', default=9463, type=int)
    parser.add_argument('--images_per_eval_segment', default=12600, type=int)


    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # check dataset
    if args.dataset_file == "swig":
        from datasets.swig import collater, processed_collater
    else:
        assert False, f"dataset {args.dataset_file} is not supported now"

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)
    # build Encoder Transformer model
    encoder_model, encoder_criterion = build_encoder_model(args)
    encoder_model.to(device)
    encoder_model_without_ddp = encoder_model
    if args.load_saved_encoder == True:
        encoder_checkpoint = torch.load(args.encoder_saved_model, map_location='cpu')
        encoder_model.load_state_dict(encoder_checkpoint["encoder_model"])
    if args.distributed:
        encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model, device_ids=[args.gpu])
        encoder_model_without_ddp = encoder_model.module
    num_encoder_parameters = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
    print('number of Encoder Transformer parameters:', num_encoder_parameters)
    encoder_param_dicts = [
        {"params": [p for n, p in encoder_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in encoder_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    encoder_optimizer = torch.optim.AdamW(encoder_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, args.lr_drop)
    
    if args.load_saved_encoder == True:
        encoder_optimizer.load_state_dict(encoder_checkpoint["encoder_optimizer"])
        encoder_lr_scheduler.load_state_dict(encoder_checkpoint["encoder_lr_scheduler"])
        args.encoder_start_epoch = encoder_checkpoint["encoder_epoch"] + 1

    # build Decoder Transformer Model
    decoder_model, decoder_criterion = build_decoder_model(args)
    decoder_model.to(device)
    decoder_model_without_ddp = decoder_model
    if args.load_saved_decoder == True:
        decoder_checkpoint = torch.load(args.decoder_saved_model, map_location='cpu')
        decoder_model.load_state_dict(decoder_checkpoint["decoder_model"])
    if args.distributed:
        decoder_model = torch.nn.parallel.DistributedDataParallel(decoder_model, device_ids=[args.gpu])
        decoder_model_without_ddp = decoder_model.module
    num_decoder_parameters = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
    print('number of Decoder Transformer parameters:', num_decoder_parameters)
    decoder_param_dicts = [
        {"params": [p for n, p in decoder_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in decoder_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    decoder_optimizer = torch.optim.AdamW(decoder_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, args.lr_drop)
    if args.load_saved_decoder == True:
        """
        decoder_optimizer.load_state_dict(decoder_checkpoint["decoder_optimizer"])
        decoder_lr_scheduler.load_state_dict(decoder_checkpoint["decoder_lr_scheduler"])
        """
        args.decoder_start_epoch = decoder_checkpoint["decoder_epoch"] + 1

    # Dataset Sampler
    # For Encoder Transformer
    if not args.test and not args.dev:
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.dev:
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        elif args.test:
            if args.distributed:
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    # For preprocessing
    if args.preprocess == True:
        preprocess_sampler_train = torch.utils.data.RandomSampler(dataset_train)
        preprocess_sampler_val = torch.utils.data.RandomSampler(dataset_val)
        preprocess_sampler_test = torch.utils.data.RandomSampler(dataset_test)

    output_dir = Path(args.output_dir)
    # dataset loader
    # For Encoder Transformer
    if not args.test and not args.dev:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.encoder_batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                    collate_fn=collater, batch_sampler=batch_sampler_train)
        data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                    drop_last=False, collate_fn=collater, sampler=sampler_val)
    else:
        if args.dev:
            data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                        drop_last=False, collate_fn=collater, sampler=sampler_val)
        elif args.test:
            data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers,
                                        drop_last=False, collate_fn=collater, sampler=sampler_test)
    # For Preprocessing
    if args.preprocess == True:
        batch_preprocess_sampler_train = torch.utils.data.BatchSampler(preprocess_sampler_train, args.encoder_batch_size, drop_last=False)
        batch_preprocess_sampler_val = torch.utils.data.BatchSampler(preprocess_sampler_val, args.encoder_batch_size, drop_last=False)
        batch_preprocess_sampler_test = torch.utils.data.BatchSampler(preprocess_sampler_test, args.encoder_batch_size, drop_last=False)
        
        preprocess_data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_train)
        preprocess_data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_val)
        preprocess_data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers, drop_last=False,
                                                  collate_fn=collater, batch_sampler=batch_preprocess_sampler_test)


    # use saved model for evaluation (using dev set or test set)
    if args.dev or args.test:
        encoder_checkpoint = torch.load(args.encoder_saved_model, map_location='cpu')
        encoder_model.load_state_dict(encoder_checkpoint["encoder_model"])
        decoder_checkpoint = torch.load(args.decoder_saved_model, map_location='cpu')
        decoder_model.load_state_dict(decoder_checkpoint["decoder_model"])
        # build dataset
        if args.dev:
            with open("__storage__/valDict.json") as val_json:
                val_dict = json.load(val_json)
            processed_dataset_val = build_processed_dataset(image_set='val', args=args, neighbors_dict=val_dict)
            if args.distributed:
                sampler_val = DistributedSampler(processed_dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(processed_dataset_val)
            data_loader = DataLoader(processed_dataset_val, num_workers=args.num_workers,
                                     drop_last=False, collate_fn=processed_collater,
                                     sampler=sampler_val)
        elif args.test:
            with open("__storage__/testDict.json") as test_json:
                test_dict = json.load(test_json)
            processed_dataset_test = build_processed_dataset(image_set='test', args=args, neighbors_dict=test_dict)
            if args.distributed:
                sampler_test = DistributedSampler(processed_dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(processed_dataset_test)
            data_loader = DataLoader(processed_dataset_test, num_workers=args.num_workers,
                                     drop_last=False, collate_fn=processed_collater,
                                     sampler=sampler_test)
            

        test_stats = evaluate_swig(encoder_model, decoder_model, decoder_criterion,
                                   data_loader, device, args.output_dir)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return None
    
    if not os.path.exists("__storage__"):
        os.mkdir("__storage__")
    # train GSRFormer Encoder Transformer
    print("Start training GSRFormer Encoder Transformer at epoch ",args.encoder_start_epoch)
    start_time = time.time()
    max_test_verb_acc_top1 = 4
    for epoch in range(args.encoder_start_epoch, args.encoder_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = encoder_train_one_epoch(encoder_model, encoder_criterion, data_loader_train, encoder_optimizer, 
                                      device, epoch, args.clip_max_norm)
        encoder_lr_scheduler.step()

        # evaluate
        test_stats = encoder_evaluate_swig(encoder_model, encoder_criterion, data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'encoder_epoch': epoch,
                     'num_encoder_parameters': num_encoder_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'encoder_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'encoder_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'encoder_model': encoder_model_without_ddp.state_dict(),
                                      'encoder_optimizer': encoder_optimizer.state_dict(),
                                      'encoder_lr_scheduler': encoder_lr_scheduler.state_dict(),
                                      'encoder_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Preprocess
    torch.cuda.empty_cache()
    if args.preprocess == True:
        preprocess_neighbors(encoder_model, preprocess_data_loader_train,
                             "train", device, args.images_per_segment)
        preprocess_neighbors(encoder_model, preprocess_data_loader_val,
                             "val", device, args.images_per_eval_segment)
        preprocess_neighbors(encoder_model, preprocess_data_loader_test,
                             "test", device, args.images_per_eval_segment)
    if args.decoder_start_epoch >= args.decoder_epochs:
        return None
    
    # build Decoder Transformer dataset
    with open("__storage__/trainDict.json") as train_json:
        train_dict = json.load(train_json)
    with open("__storage__/valDict.json") as val_json:
        val_dict = json.load(val_json)
    processed_dataset_train = build_processed_dataset("train", args, neighbors_dict=train_dict)
    processed_dataset_val = build_processed_dataset("val", args, neighbors_dict=val_dict)
    # build Decoder Transformer dataset sampler
    if args.distributed:
        sampler_train = DistributedSampler(processed_dataset_train)
        sampler_val = DistributedSampler(processed_dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(processed_dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(processed_dataset_val)
    
    # build Decoder Transformer dataset loader
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.decoder_batch_size, drop_last=True)
    data_loader_train = DataLoader(processed_dataset_train, num_workers=args.num_workers,
                                   collate_fn=processed_collater, batch_sampler=batch_sampler_train)
    data_loader_val = DataLoader(processed_dataset_val, num_workers=args.num_workers,
                                 drop_last=False, collate_fn=processed_collater, sampler=sampler_val)

    # Train GSRFormer Decoder Transformer
    print("Start training GSRFormer Decoder Transformer at epoch ",args.decoder_start_epoch)
    start_time = time.time()
    max_test_verb_acc_top1 = 43
    for epoch in range(args.decoder_start_epoch, args.decoder_epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = decoder_train_one_epoch(decoder_model, decoder_criterion, encoder_model,
                                              data_loader_train, decoder_optimizer, 
                                              device, epoch, args.images_per_segment,
                                              args.clip_max_norm)
        decoder_lr_scheduler.step()

        # evaluate
        test_stats = evaluate_swig(encoder_model, decoder_model, decoder_criterion,
                                           data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'decoder_epoch': epoch,
                     'num_decoder_parameters': num_decoder_parameters} 
        if args.output_dir:
            checkpoint_paths = [output_dir / 'decoder_checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_verb_acc_top1_unscaled'] > max_test_verb_acc_top1:
                max_test_verb_acc_top1 = log_stats['test_verb_acc_top1_unscaled']
                checkpoint_paths.append(output_dir / f'decoder_checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'decoder_model': decoder_model_without_ddp.state_dict(),
                                      'decoder_optimizer': decoder_optimizer.state_dict(),
                                      'decoder_lr_scheduler': decoder_lr_scheduler.state_dict(),
                                      'decoder_epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GSRFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
