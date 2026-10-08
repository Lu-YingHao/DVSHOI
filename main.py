"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
from datetime import timedelta
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from upt import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory, validate_training_checkpoint

warnings.filterwarnings("ignore")

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes)
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    dvs_root = resolve_dvs_root(args)
    dvs_sensor_size = tuple(args.dvs_sensor_size)

    trainset = DataFactory(
        name=args.dataset,
        partition=args.partitions[0],
        data_root=args.data_root,
        dvs_root=dvs_root,
        dvs_sensor_size=dvs_sensor_size,
        dvs_num_bins=args.dvs_num_bins,
    )
    testset = DataFactory(
        name=args.dataset,
        partition=args.partitions[1],
        data_root=args.data_root,
        dvs_root=dvs_root,
        dvs_sensor_size=dvs_sensor_size,
        dvs_num_bins=args.dvs_num_bins,
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    args.human_idx = 0
    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_classes = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_classes = 24
    
    upt = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: load model from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        inference_loader=test_loader if args.eval_last_epochs else None,
        eval_last_epochs=args.eval_last_epochs,
        vcoco_eval_root=args.vcoco_eval_root,
        vcoco_exclude_actions=args.vcoco_exclude_actions,
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            if rank == 0:
                from vcoco.evaluation import evaluate_vcoco_cache, load_evaluator
                engine.cache_vcoco(test_loader, args.output_dir, net=engine._state.net.module)
                evaluator = load_evaluator(
                    args.vcoco_eval_root, args.partitions[1], args.vcoco_exclude_actions)
                metrics = evaluate_vcoco_cache(
                    evaluator, os.path.join(args.output_dir, 'cache.pkl'))
                print('[V-COCO METRICS]', metrics)
            return
        ap = engine.test_hico(test_loader)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean():.4f},"
            f" rare: {ap[rare].mean():.4f},"
            f" none-rare: {ap[non_rare].mean():.4f}"
        )
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
        if "interaction_head" in n and p.requires_grad]
    }]
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    if args.resume_training:
        engine.restore_training_state(checkpoint, args.epochs)

    engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(
        name='hicodet',
        partition=args.partitions[0],
        data_root=args.data_root,
        dvs_root=resolve_dvs_root(args),
        dvs_sensor_size=tuple(args.dvs_sensor_size),
        dvs_num_bins=args.dvs_num_bins,
    )
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

def resolve_dvs_root(args):
    if not args.use_dvs:
        return None
    if args.dvs_root:
        return os.path.expanduser(args.dvs_root)

    dataset_dirs = {
        'hicodet': 'hico-dvs',
        'vcoco': 'vcoco-dvs',
    }
    if args.dataset not in dataset_dirs:
        raise ValueError("Unknown dataset " + args.dataset)

    env_root = os.environ.get('DVS_DATA_ROOT')
    if env_root:
        return os.path.expanduser(env_root)

    dataset_dir = dataset_dirs[args.dataset]
    data_roots = [
        '/media/think/disk2/lyh/Data',
        os.path.expanduser('~/Data'),
    ]
    for data_root in data_roots:
        dvs_root = os.path.join(data_root, dataset_dir)
        if os.path.isdir(dvs_root):
            return dvs_root

    return os.path.join(data_roots[-1], dataset_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--use-dvs', action='store_true')
    parser.add_argument('--dvs-root', default='', type=str)
    parser.add_argument('--dvs-sensor-size', nargs=2, default=[260, 346], type=int)
    parser.add_argument('--dvs-num-bins', default=8, type=int)
    parser.add_argument('--dvs-variant', default='base', choices=('tiny', 'base'))

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--resume-training', action='store_true',
                        help='Restore optimizer, scheduler and counters; --epochs is the final epoch')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--eval-last-epochs', default=0, type=int,
                        help='Synchronously evaluate the final N V-COCO training epochs')
    parser.add_argument('--vcoco-eval-root', default=None,
                        help='External s-gupta/v-coco tools and official annotations')
    parser.add_argument('--vcoco-exclude-actions', nargs='*', default=['point'],
                        help='Explicit evaluation exclusions; pass no values for all actions')
    parser.add_argument('--distributed-timeout-minutes', default=180, type=int,
                        help='Collective timeout, including rank-zero synchronous inference')
    parser.add_argument('--check-only', action='store_true',
                        help='Check dataset pairing and evaluation resources without training')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    args = parser.parse_args()
    if args.epochs < 1 or args.eval_last_epochs < 0:
        parser.error('--epochs must be positive and --eval-last-epochs nonnegative')
    if len(args.partitions) != 2:
        parser.error('--partitions requires a training split and an inference split')
    if args.eval_last_epochs and args.dataset != 'vcoco':
        parser.error('--eval-last-epochs currently supports --dataset vcoco')
    if args.distributed_timeout_minutes <= 0:
        parser.error('--distributed-timeout-minutes must be positive')
    if args.resume_training:
        if not os.path.isfile(args.resume):
            parser.error('--resume-training requires an existing --resume checkpoint')
        if args.eval or args.cache or args.sanity:
            parser.error('--resume-training is only for training or --check-only')
    if args.dataset == 'vcoco' and (args.eval or args.cache) and not os.path.isfile(args.resume):
        parser.error('V-COCO --eval/--cache requires an existing --resume checkpoint')
    print(args)

    if args.dataset == 'vcoco' and (args.eval_last_epochs or args.eval or args.check_only):
        from vcoco.evaluation import load_evaluator, resolve_evaluation_root
        args.vcoco_eval_root = resolve_evaluation_root(args.vcoco_eval_root)
        # Fail before a long training job if external annotations/tools are missing.
        load_evaluator(args.vcoco_eval_root, args.partitions[1], args.vcoco_exclude_actions)

    if args.check_only:
        start_epoch = 0
        if args.resume_training:
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = validate_training_checkpoint(checkpoint, args.epochs)
        for split in args.partitions:
            dataset = DataFactory(
                args.dataset, split, args.data_root,
                dvs_root=resolve_dvs_root(args),
                dvs_sensor_size=tuple(args.dvs_sensor_size), dvs_num_bins=args.dvs_num_bins)
            if args.resume_training and split == args.partitions[0]:
                iterations = (len(dataset) + args.world_size - 1) // args.world_size // args.batch_size
                validate_training_checkpoint(checkpoint, args.epochs, iterations)
            for index in range(len(dataset)):
                filename = dataset.dataset.filename(index)
                rgb_path = os.path.join(dataset.dataset._root, filename)
                if not os.path.isfile(rgb_path):
                    raise FileNotFoundError('Missing RGB image: ' + rgb_path)
                if args.use_dvs and not os.path.isfile(dataset._dvs_path(index)):
                    raise FileNotFoundError('Missing DVS events: ' + dataset._dvs_path(index))
            image, target = dataset[0]
            print('[CHECK] split={} samples={} rgb_shape={} dvs_shape={}'.format(
                split, len(dataset), tuple(image.shape),
                tuple(target['dvs_frames'].shape) if args.use_dvs else None))
        if args.pretrained and not os.path.isfile(args.pretrained):
            raise FileNotFoundError('Missing pretrained detector: ' + args.pretrained)
        print('[CHECK] CUDA available={}, GPU count={}'.format(
            torch.cuda.is_available(), torch.cuda.device_count()))
        if args.resume_training:
            print('[CHECK] resume epoch {} -> {}; optimizer and scheduler will be restored'.format(
                start_epoch, args.epochs))
        if args.eval_last_epochs:
            print('[CHECK] synchronous inference epochs: {} through {}'.format(
                max(start_epoch + 1, args.epochs - args.eval_last_epochs + 1), args.epochs))
        sys.exit()

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
