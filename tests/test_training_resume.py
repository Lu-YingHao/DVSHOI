import copy
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from utils import CustomisedDLE, validate_training_checkpoint


class TestTrainingResume(unittest.TestCase):
    def make_engine(self, epoch=0):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
        engine = object.__new__(CustomisedDLE)
        engine._device = torch.device('cpu')
        engine._rank = 1
        engine._train_loader = [(torch.ones(1, 2), [])]
        engine._state = SimpleNamespace(
            net=model, optimizer=optimizer, lr_scheduler=scheduler,
            scaler=Mock(), epoch=epoch, iteration=epoch,
            t_data=[], t_iteration=[], running_loss=[],
        )
        return engine

    def make_checkpoint(self, epoch):
        engine = self.make_engine()
        for _ in range(epoch):
            engine._state.optimizer.zero_grad()
            engine._state.net(torch.ones(1, 2)).sum().backward()
            engine._state.optimizer.step()
            saved = copy.deepcopy(dict(
                epoch=epoch, iteration=epoch,
                model_state_dict=engine._state.net.state_dict(),
                optim_state_dict=engine._state.optimizer.state_dict(),
                scheduler_state_dict=engine._state.lr_scheduler.state_dict(),
                scaler_state_dict={},
            ))
            engine._state.lr_scheduler.step()
        return saved, engine

    def test_optimizer_moments_and_pending_scheduler_step_are_restored(self):
        # Epoch 10 checks the actual LR drop; epoch 12 checks this user's run.
        for epoch in (10, 12):
            checkpoint, uninterrupted = self.make_checkpoint(epoch)
            resumed = self.make_engine()
            resumed._state.net.load_state_dict(checkpoint['model_state_dict'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                resumed.restore_training_state(checkpoint, 16)
            self.assertEqual(resumed._state.epoch, epoch)
            self.assertEqual(resumed._state.iteration, epoch)
            self.assertEqual(resumed._state.lr_scheduler.state_dict(),
                             uninterrupted._state.lr_scheduler.state_dict())
            self.assertEqual(resumed._state.optimizer.param_groups[0]['lr'], 1e-5)
            resumed._state.scaler.load_state_dict.assert_called_once_with({})
            # Identical next update demonstrates restored Adam moments, not
            # just restored LR and model weights.
            for engine in (resumed, uninterrupted):
                engine._state.optimizer.zero_grad()
                engine._state.net(torch.ones(1, 2)).sum().backward()
                engine._state.optimizer.step()
            for actual, expected in zip(resumed._state.net.parameters(),
                                        uninterrupted._state.net.parameters()):
                torch.testing.assert_allclose(actual, expected)

    def test_already_stepped_checkpoint_does_not_step_twice(self):
        checkpoint, original = self.make_checkpoint(12)
        checkpoint['scheduler_state_dict'] = original._state.lr_scheduler.state_dict()
        resumed = self.make_engine()
        resumed.restore_training_state(checkpoint, 16)
        self.assertEqual(resumed._state.lr_scheduler.last_epoch, 12)

    def test_resumed_loop_trains_and_evaluates_only_epochs_13_to_16(self):
        engine = self.make_engine(epoch=12)
        engine.inference_loader = object()
        engine.eval_last_epochs = 4
        events = []
        engine._on_start = Mock()
        engine._on_end = Mock()

        def start_epoch():
            engine._state.epoch += 1
            events.append(('train', engine._state.epoch))

        def iteration():
            engine._state.iteration += 1
            engine._state.loss = torch.tensor(1.)

        engine._on_start_epoch = start_epoch
        engine._on_start_iteration = Mock()
        engine._on_each_iteration = iteration
        engine._on_end_iteration = Mock()
        engine._run_vcoco_epoch_evaluation = lambda: events.append(('eval', engine._state.epoch))
        # Use the real checkpoint/scheduler lifecycle, with a fake file writer.
        engine._rank = 0
        engine.save_checkpoint = lambda: events.append(('save', engine._state.epoch))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            engine(16)
        self.assertEqual(events, [(kind, epoch) for epoch in range(13, 17)
                                  for kind in ('train', 'save', 'eval')])
        self.assertEqual(engine._state.iteration, 16)
        self.assertEqual(engine.epochs, 16)

    def test_fresh_training_still_runs_requested_epochs(self):
        engine = self.make_engine()
        seen = []
        engine._on_start = Mock()
        engine._on_end = Mock()
        engine._on_start_iteration = Mock()
        engine._on_end_iteration = Mock()
        engine._on_end_epoch = Mock()

        def start_epoch():
            engine._state.epoch += 1
            seen.append(engine._state.epoch)

        def iteration():
            engine._state.loss = torch.tensor(1.)

        engine._on_start_epoch = start_epoch
        engine._on_each_iteration = iteration
        engine(2)
        self.assertEqual(seen, [1, 2])

    def test_invalid_or_incompatible_checkpoint_is_rejected(self):
        checkpoint, _ = self.make_checkpoint(12)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            validate_training_checkpoint(checkpoint, 12)
        with self.assertRaisesRegex(ValueError, 'batch size'):
            validate_training_checkpoint(checkpoint, 16, 2)
        checkpoint.pop('optim_state_dict')
        with self.assertRaisesRegex(ValueError, 'optim_state_dict'):
            validate_training_checkpoint(checkpoint, 16)


if __name__ == '__main__':
    unittest.main()
