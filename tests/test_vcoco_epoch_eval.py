import contextlib
import csv
import io
import json
import os
import pickle
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from utils import CacheTemplate, CustomisedDLE
from vcoco.evaluation import evaluate_vcoco_cache, load_evaluator, resolve_evaluation_root


class TestEpochEvaluation(unittest.TestCase):
    def test_final_four_epochs_save_before_evaluation(self):
        for total, expected in [(20, [17, 18, 19, 20]), (12, [9, 10, 11, 12]),
                                (2, [1, 2])]:
            engine = object.__new__(CustomisedDLE)
            engine.epochs = total
            engine.eval_last_epochs = 4
            engine.inference_loader = object()
            engine._state = SimpleNamespace(epoch=0)
            events = []
            engine._run_vcoco_epoch_evaluation = lambda: events.append(
                ('eval', engine._state.epoch))
            with patch('utils.DistributedLearningEngine._on_end_epoch',
                       side_effect=lambda: events.append(('save', engine._state.epoch))):
                for epoch in range(1, total + 1):
                    engine._state.epoch = epoch
                    engine._on_end_epoch()
            self.assertEqual([epoch for kind, epoch in events if kind == 'eval'], expected)
            for epoch in expected:
                index = events.index(('eval', epoch))
                self.assertEqual(events[index - 1], ('save', epoch))

    def test_disabled_evaluation(self):
        engine = object.__new__(CustomisedDLE)
        engine.epochs = 20
        engine.eval_last_epochs = 0
        engine.inference_loader = object()
        engine._state = SimpleNamespace(epoch=20)
        engine._run_vcoco_epoch_evaluation = Mock()
        with patch('utils.DistributedLearningEngine._on_end_epoch'):
            engine._on_end_epoch()
        engine._run_vcoco_epoch_evaluation.assert_not_called()

    def test_nonmaster_waits_without_running_inference(self):
        engine = object.__new__(CustomisedDLE)
        engine._rank = 1
        engine._device = torch.device('cpu')
        engine._evaluate_vcoco_epoch = Mock()
        with patch('utils.dist.barrier') as barrier, patch('utils.dist.broadcast') as broadcast:
            engine._run_vcoco_epoch_evaluation()
        barrier.assert_called_once()
        broadcast.assert_called_once()
        engine._evaluate_vcoco_epoch.assert_not_called()

    def test_master_failure_is_broadcast_to_other_ranks(self):
        engine = object.__new__(CustomisedDLE)
        engine._rank = 0
        engine._device = torch.device('cpu')
        engine._evaluate_vcoco_epoch = Mock(side_effect=ValueError('evaluation failed'))
        sent = []
        with patch('utils.dist.barrier'), \
                patch('utils.dist.broadcast', side_effect=lambda value, src: sent.append(value.item())), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                engine._run_vcoco_epoch_evaluation()
        self.assertEqual(sent, [0])

    def test_metrics_history_and_training_state_are_restored(self):
        with tempfile.TemporaryDirectory() as root:
            model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
            model.train()
            model[1].eval()
            engine = object.__new__(CustomisedDLE)
            engine._state = SimpleNamespace(epoch=17, iteration=5,
                                            net=SimpleNamespace(module=model))
            engine._device = torch.device('cpu')
            engine._cache_dir = root
            engine.inference_loader = SimpleNamespace(dataset=SimpleNamespace(partition='test'))
            engine.vcoco_eval_root = root
            engine.vcoco_exclude_actions = ('point',)
            engine._vcoco_evaluator = None

            def cache(loader, output_dir, net):
                self.assertIs(net, model)
                net.eval()
                random.random()
                np.random.rand()
                torch.rand(1)
                with open(os.path.join(output_dir, 'cache.pkl'), 'wb') as file:
                    pickle.dump([], file)

            engine.cache_vcoco = cache
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            metrics = dict(agent_map=61.2, scenario1_role_map=59.3, scenario2_role_map=65.4)
            with patch('torch.cuda.get_rng_state', return_value=torch.get_rng_state()), \
                    patch('torch.cuda.set_rng_state'), \
                    patch('vcoco.evaluation.load_evaluator', return_value=object()) as load, \
                    patch('vcoco.evaluation.evaluate_vcoco_cache', return_value=metrics), \
                    contextlib.redirect_stdout(io.StringIO()):
                for epoch in (17, 18):
                    engine._state.epoch = epoch
                    Path(root, 'ckpt_00005_{:02d}.pt'.format(epoch)).touch()
                    engine._evaluate_vcoco_epoch()
            load.assert_called_once_with(root, 'test', ('point',))
            self.assertTrue(model.training)
            self.assertFalse(model[1].training)
            self.assertEqual(random.getstate(), rng[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1], rng[1][1]))
            self.assertEqual(np.random.get_state()[2:], rng[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))
            with open(os.path.join(root, 'vcoco_metrics.csv')) as file:
                records = list(csv.DictReader(file))
            self.assertEqual([row['epoch'] for row in records], ['17', '18'])
            with open(os.path.join(root, 'epoch_18_eval', 'metrics.json')) as file:
                record = json.load(file)
            self.assertEqual(record['excluded_actions'], ['point'])
            self.assertEqual(record['scenario1_role_map'], 59.3)


class TestVCOCOProtocol(unittest.TestCase):
    def test_print_only_evaluator_metrics_are_captured(self):
        class Evaluator:
            def _do_eval(self, path, ovr_thresh):
                print('Average Agent AP = 61.25')
                print('Average Role [scenario_1] AP = 59.50')
                print('Average Role [scenario_2] AP = 65.75')

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_vcoco_cache(Evaluator(), 'unused.pkl')
        self.assertEqual(metrics, dict(agent_map=61.25, scenario1_role_map=59.5,
                                       scenario2_role_map=65.75))

    def test_upstream_constructor_filters_point_alias(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ['data/vcoco/vcoco_test.json', 'data/instances_vcoco_all_2014.json',
                         'data/splits/vcoco_test.ids']:
                target = Path(root, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('[]')
            Path(root, 'vsrl_eval.py').write_text(
                'class VCOCOeval:\n'
                '    def __init__(self, *paths):\n'
                "        self.VCOCO = [{'action_name': 'hold'}, {'action_name': 'point'}]\n"
                '        self._init_vcoco()\n'
                '    def _init_vcoco(self):\n'
                "        self.actions = [item['action_name'] for item in self.VCOCO]\n")
            self.assertEqual(load_evaluator(root, exclude_actions=('points',)).actions, ['hold'])
            self.assertEqual(load_evaluator(root, exclude_actions=()).actions, ['hold', 'point'])

    def test_real_evaluator_exclusion_changes_ap_denominator(self):
        try:
            tools = resolve_evaluation_root()
        except FileNotFoundError:
            self.skipTest('External V-COCO evaluator is not installed')
        with tempfile.TemporaryDirectory() as root:
            shutil.copyfile(os.path.join(tools, 'vsrl_eval.py'), os.path.join(root, 'vsrl_eval.py'))
            for folder in ['data/vcoco', 'data/splits']:
                Path(root, folder).mkdir(parents=True)
            images = [dict(id=i, width=100, height=100) for i in (1, 2)]
            annotations = []
            for i in (1, 2):
                annotations.extend([
                    dict(id=i * 10, image_id=i, category_id=1, bbox=[10, 10, 20, 30],
                         area=600, iscrowd=0),
                    dict(id=i * 10 + 1, image_id=i, category_id=2, bbox=[40, 10, 10, 10],
                         area=100, iscrowd=0)])
            coco = dict(images=images, annotations=annotations,
                        categories=[dict(id=1, name='person'), dict(id=2, name='object')])
            Path(root, 'data/instances_vcoco_all_2014.json').write_text(json.dumps(coco))
            actions = [dict(action_name=action, role_name=['agent', 'obj'], image_id=[1, 2],
                            ann_id=[10, 20], label=[1, 1], role_object_id=[10, 20, 11, 21])
                       for action in ('hold', 'point')]
            Path(root, 'data/vcoco/vcoco_test.json').write_text(json.dumps(actions))
            Path(root, 'data/splits/vcoco_test.ids').write_text('1\n2\n')
            predictions = [CacheTemplate(image_id=i, person_box=[10, 10, 29, 39],
                                         hold_agent=0.9, hold_obj=[40, 10, 49, 19, 0.9],
                                         point_agent=0.9, point_obj=[70, 70, 79, 79, 0.9])
                           for i in (1, 2)]
            cache = os.path.join(root, 'cache.pkl')
            with open(cache, 'wb') as file:
                pickle.dump(predictions, file, protocol=2)
            with contextlib.redirect_stdout(io.StringIO()):
                excluded = load_evaluator(root, exclude_actions=('points',))
                included = load_evaluator(root, exclude_actions=())
                excluded_metrics = evaluate_vcoco_cache(excluded, cache)
                included_metrics = evaluate_vcoco_cache(included, cache)
            self.assertNotIn('point', excluded.actions)
            self.assertEqual(excluded_metrics['scenario1_role_map'], 100.0)
            self.assertEqual(excluded_metrics['scenario2_role_map'], 100.0)
            self.assertEqual(included_metrics['scenario1_role_map'], 50.0)
            self.assertEqual(included_metrics['scenario2_role_map'], 50.0)


if __name__ == '__main__':
    unittest.main()
