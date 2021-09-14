# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.ag.action_genome_eval import ActionGenomeEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument("--eval_span", default=5, type=int)

    # * Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # * Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # * Dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--ag_path", type=str)
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # * Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="whether distributed process or not")
    parser.add_argument(
        "--port_num", default=29500, type=int, help="port number when the process is multi node"
    )

    # * Multi node distributed training parameters
    parser.add_argument("--multi_node", action="store_true", help="whether multi node process or not")
    parser.add_argument("--hostfile", default="./hostfile", type=str, help="path of hostfile")
    return parser


def setup_distributed(args, local_rank):
    args.local_rank = local_rank

    if args.multi_node:
        current_dir = os.getcwd()
        with open(args.hostfile) as f:
            host = f.readlines()
        host[0] = host[0].rstrip("\n")
        args.dist_url = f"tcp://{host[0]}:{args.port_num}"
        args.rank = args.ngpus * args.node_rank + args.local_rank
        args.world_size = args.ngpus * args.node_size
    else:
        args.dist_url = f"tcp://127.0.0.1:{args.port_num}"
        args.rank = local_rank
        args.world_size = args.ngpus * args.node_size

    torch.cuda.set_device(args.local_rank)

    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}) / (world_size {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # Disables printing when not in master process
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print_only_master(*_args, **kwargs):
        force = kwargs.pop("force", False)
        if args.rank == 0 or force:
            builtin_print(*_args, **kwargs)

    __builtin__.print = print_only_master

    # To resolve "Too many open files" error.
    mp.set_sharing_strategy("file_system")


def _main(local_rank, args):
    if args.distributed:
        setup_distributed(args, local_rank)

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup model, criterion, postprocessors.
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Setup distributed model.
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Setup optimizer.
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Setup dataset.
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    # Setup distributed dataset.
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    # Setup dataloader.
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    # Setup dataset file for evaluation.
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    # Setup resume.
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    # Setup evaluator.
    if args.dataset_file == "ag":
        evaluator = ActionGenomeEvaluator(device)
    elif args.dataset_file == "coco":
        iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
        evaluator = CocoEvaluator(base_ds, iou_types)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    else:
        evaluator = PanopticEvaluator(
            data_loader_val.dataset.ann_file,
            data_loader_val.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # Execute evaluation only.
    if args.eval:
        print("Start evaluation")
        test_stats = evaluate(
            evaluator,
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
        )
        if args.output_dir:
            if isinstance(evaluator, CocoEvaluator):
                utils.save_on_master(evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # Execute training.
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = {"eval": "Skip"}
        if (epoch + 1) % args.eval_span == 0:
            test_stats = evaluate(
                evaluator,
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if isinstance(evaluator, ActionGenomeEvaluator):
                (output_dir / "eval").mkdir(exist_ok=True)
                filenames = ["latest.pth"]
                if epoch % 50 == 0:
                    filenames.append(f"{epoch:03}.pth")
                for name in filenames:
                    torch.save(evaluator.eval, output_dir / "eval" / name)
            elif isinstance(evaluator, CocoEvaluator):
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def cleanup(args):
    if args.distributed:
        dist.destroy_process_group()


def main(local_rank, args):
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    try:
        _main(local_rank, args)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        cleanup(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        if args.multi_node and "OMPI_COMM_WORLD_RANK" in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
            print("Multi node distributed mode")
            args.node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            args.node_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            args.ngpus = torch.cuda.device_count()
        else:
            print("Single node distributed mode")
            args.node_rank = 0
            args.node_size = 1
            args.ngpus = torch.cuda.device_count()

        print("node_rank: {}".format(args.node_rank))
        print("node_size: {}".format(args.node_size))
        print("ngpus: {}".format(args.ngpus))
    else:
        print("Not distributed")

    mp.spawn(main, nprocs=args.ngpus, args=(args,))
