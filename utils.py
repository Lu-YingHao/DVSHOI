"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import contextlib
import csv
import json
import random
import sys
import time
import traceback
import torch
import torch.distributed as dist
import pickle
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket.pocket as pocket
from pocket.pocket.core import DistributedLearningEngine
from pocket.pocket.utils import DetectionAPMeter, BoxPairAssociation

sys.path.append('detr')
import datasets.transforms as T
from dvs_data import events_to_frames, load_event_npz

def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(
        self,
        name,
        partition,
        data_root,
        dvs_root=None,
        dvs_sensor_size=(260, 346),
        dvs_num_bins=8,
    ):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            image_root = os.path.join(data_root, image_dir[partition])
            if not os.path.isdir(image_root):
                fallback_dir = 'val2014' if partition == 'test' else 'train2014'
                image_root = os.path.join(data_root, 'v_coco/images', fallback_dir)
            self.dataset = VCOCO(
                root=image_root,
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        self.name = name
        self.partition = partition
        self.dvs_root = self._resolve_dvs_root(dvs_root)
        self.dvs_sensor_size = tuple(dvs_sensor_size) if dvs_sensor_size else None
        self.dvs_num_bins = int(dvs_num_bins)
        if self.dvs_num_bins <= 0:
            raise ValueError("dvs_num_bins must be positive")

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

    def __len__(self):
        return len(self.dataset)

    def _resolve_dvs_root(self, dvs_root):
        if dvs_root is None:
            return None

        if self.name == 'hicodet':
            split = self.partition
        else:
            split = 'test' if self.partition == 'test' else 'train'

        split_root = os.path.join(dvs_root, split)
        return split_root if os.path.isdir(split_root) else dvs_root

    def _dvs_path(self, index):
        filename = self.dataset.filename(index)
        stem = os.path.splitext(filename)[0]
        return os.path.join(self.dvs_root, stem + '.npz')

    def _attach_dvs(self, index, image, target):
        filename = self.dataset.filename(index)
        dvs_path = self._dvs_path(index)
        events = load_event_npz(dvs_path, sensor_size=self.dvs_sensor_size)

        image_width, image_height = image.size
        target['dvs'] = events
        target['dvs_frames'] = events_to_frames(
            events, num_bins=self.dvs_num_bins
        )
        target['dvs_path'] = dvs_path
        target['file_name'] = filename
        target['rgb_size'] = torch.as_tensor(
            [image_height, image_width],
            dtype=torch.int64
        )
        target['dvs_sensor_size'] = torch.as_tensor(
            events['sensor_size'],
            dtype=torch.int64
        )
        return target

    def __getitem__(self, i):
        image, target = self.dataset[i]

        if self.dvs_root is not None:
            target = self._attach_dvs(i, image, target)

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)

        return image, target

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

def validate_training_checkpoint(checkpoint, total_epochs, iterations_per_epoch=None):
    required = ('epoch', 'iteration', 'model_state_dict',
                'optim_state_dict', 'scheduler_state_dict')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError('Training checkpoint is missing: ' + ', '.join(missing))
    epoch, iteration = checkpoint['epoch'], checkpoint['iteration']
    if not isinstance(epoch, int) or epoch < 0 or epoch >= total_epochs:
        raise ValueError('Target --epochs must exceed the saved epoch ({})'.format(epoch))
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError('Invalid checkpoint iteration')
    if iterations_per_epoch is not None and iteration != epoch * iterations_per_epoch:
        raise ValueError('Resume requires the same training split, batch size and world size; '
                         'saved iteration={} but expected {}'.format(
                             iteration, epoch * iterations_per_epoch))
    if checkpoint['scheduler_state_dict'].get('last_epoch') not in (epoch - 1, epoch):
        raise ValueError('Checkpoint scheduler epoch does not match the training epoch')
    return epoch


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117,
                 inference_loader=None, eval_last_epochs=0,
                 vcoco_eval_root=None, vcoco_exclude_actions=('point',), **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.inference_loader = inference_loader
        self.eval_last_epochs = eval_last_epochs
        self.vcoco_eval_root = vcoco_eval_root
        self.vcoco_exclude_actions = tuple(vcoco_exclude_actions)
        self._vcoco_evaluator = None

    def restore_training_state(self, checkpoint, total_epochs):
        epoch = validate_training_checkpoint(
            checkpoint, total_epochs, len(self._train_loader))
        self._state.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        for parameter, state in self._state.optimizer.state.items():
            self._state.optimizer.state[parameter] = self._move_to_device(state)
        self._state.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Pocket saves the checkpoint BEFORE stepping the epoch scheduler.
        # Replay that pending step, including at a learning-rate drop boundary.
        if self._state.lr_scheduler.last_epoch == epoch - 1:
            self._state.lr_scheduler.step()
        if 'scaler_state_dict' in checkpoint:
            self._state.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self._state.epoch = epoch
        self._state.iteration = checkpoint['iteration']
        if self._rank == 0:
            print('[RESUME] completed_epoch={} next_epoch={} target_epoch={} lr={}'.format(
                epoch, epoch + 1, total_epochs,
                [group['lr'] for group in self._state.optimizer.param_groups]))

    def __call__(self, n):
        # Treat n as the final epoch, preserving cumulative checkpoint numbering
        # and the final-N evaluation schedule when resuming an earlier run.
        if self._state.epoch >= n:
            raise ValueError('Target epochs must exceed the completed epoch')
        self.epochs = n
        self._on_start()
        for _ in range(self._state.epoch, n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.inputs = batch[:-1]
                self._state.targets = batch[-1]
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)
                self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()
            self._on_end_epoch()
        self._on_end()

    def _on_end_epoch(self):
        # Preserve the existing checkpoint and learning-rate lifecycle.
        super()._on_end_epoch()
        first_eval_epoch = max(1, self.epochs - self.eval_last_epochs + 1)
        if (self.inference_loader is not None and self.eval_last_epochs > 0
                and self._state.epoch >= first_eval_epoch):
            self._run_vcoco_epoch_evaluation()

    def _run_vcoco_epoch_evaluation(self):
        # Only rank zero runs inference. All ranks must wait before the next
        # training epoch; the unwrapped model avoids DDP forward collectives.
        dist.barrier()
        success = torch.ones(1, dtype=torch.int32, device=self._device)
        error = None
        if self._rank == 0:
            try:
                self._evaluate_vcoco_epoch()
            except Exception as exc:
                error = exc
                success.zero_()
                traceback.print_exc()
        dist.broadcast(success, src=0)
        if not success.item():
            raise RuntimeError('V-COCO epoch evaluation failed on rank zero') from error

    def _evaluate_vcoco_epoch(self):
        from vcoco.evaluation import _Tee, evaluate_vcoco_cache, load_evaluator

        epoch = self._state.epoch
        output_dir = os.path.join(self._cache_dir, 'epoch_{:02d}_eval'.format(epoch))
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = os.path.join(
            self._cache_dir, 'ckpt_{:05d}_{:02d}.pt'.format(self._state.iteration, epoch))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('Epoch checkpoint was not saved: ' + checkpoint)
        net = self._state.net.module
        modes = [(module, module.training) for module in net.modules()]
        # Evaluation must not perturb the random sequence used by training.
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state(self._device))
        started = time.monotonic()
        try:
            with open(os.path.join(output_dir, 'inference.log'), 'w', buffering=1) as log:
                with contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
                        contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                    print('[V-COCO EVAL] epoch={} checkpoint={}'.format(epoch, checkpoint))
                    print('[V-COCO PROTOCOL] split={} excluded_actions={}'.format(
                        self.inference_loader.dataset.partition, self.vcoco_exclude_actions))
                    self.cache_vcoco(self.inference_loader, output_dir, net=net)
                    if self._vcoco_evaluator is None:
                        self._vcoco_evaluator = load_evaluator(
                            self.vcoco_eval_root, self.inference_loader.dataset.partition,
                            self.vcoco_exclude_actions)
                    metrics = evaluate_vcoco_cache(
                        self._vcoco_evaluator, os.path.join(output_dir, 'cache.pkl'))
                    record = dict(
                        epoch=epoch, checkpoint=os.path.abspath(checkpoint),
                        split=self.inference_loader.dataset.partition,
                        excluded_actions=list(self.vcoco_exclude_actions),
                        elapsed_seconds=time.monotonic() - started, **metrics)
                    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
                        json.dump(record, file, indent=2)
                        file.write('\n')
                    history_path = os.path.join(self._cache_dir, 'vcoco_metrics.csv')
                    write_header = not os.path.isfile(history_path)
                    with open(history_path, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=list(record))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(record)
                    print('[V-COCO METRICS] ' + json.dumps(record, sort_keys=True))
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state(rng[3], self._device)

    def _move_to_device(self, x):
        if isinstance(x, torch.Tensor):
            if self._device.type == 'cuda':
                return x.cuda(self._device, non_blocking=True)
            return x.to(self._device)
        if isinstance(x, list):
            return [self._move_to_device(item) for item in x]
        if isinstance(x, tuple):
            return tuple(self._move_to_device(item) for item in x)
        if isinstance(x, dict):
            return {key: self._move_to_device(value) for key, value in x.items()}
        return x

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = self._move_to_device(self._state.inputs)
        self._state.targets = self._move_to_device(self._state.targets)

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if not torch.isfinite(loss_dict['interaction_loss']).all():
            net = getattr(self._state.net, 'module', self._state.net)
            stats = getattr(net, '_last_hoi_loss_stats', {})
            filenames = [target.get('file_name', '<unknown>')
                         for target in self._state.targets]
            raise ValueError(
                'The HOI loss is non-finite (NaN/Inf) for rank {}; '
                'epoch={} iteration={} loss={} stats={} files={}'.format(
                    self._rank, self._state.epoch, self._state.iteration,
                    loss_dict['interaction_loss'].detach().item(), stats, filenames))

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            targets = self._move_to_device(batch[-1])
            output = net(inputs, targets=targets)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            target = batch[-1][0]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                        gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                        boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            meter.append(scores, interactions, labels)

        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            targets = self._move_to_device(batch[-1])
            output = net(inputs, targets=targets)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache', net=None):
        net = self._state.net if net is None else net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            targets = self._move_to_device(batch[-1])
            output = net(inputs, targets=targets)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
