"""Run the external V-COCO evaluator with an explicit evaluation protocol."""

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import re


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def resolve_evaluation_root(root=None):
    candidates = [root] if root else [
        os.environ.get('VCOCO_EVAL_ROOT'),
        os.path.join(os.path.dirname(__file__), 'v_coco'),
        os.path.expanduser('~/Data/v-coco'),
    ]
    for candidate in candidates:
        if candidate:
            candidate = os.path.abspath(os.path.expanduser(candidate))
            if os.path.isfile(os.path.join(candidate, 'vsrl_eval.py')):
                return candidate
    raise FileNotFoundError(
        'V-COCO evaluation tools not found. Set --vcoco-eval-root to the '
        's-gupta/v-coco directory containing vsrl_eval.py and data/.')


def load_evaluator(root, split='test', exclude_actions=('point',)):
    root = resolve_evaluation_root(root)
    paths = [
        os.path.join(root, 'data', 'vcoco', 'vcoco_{}.json'.format(split)),
        os.path.join(root, 'data', 'instances_vcoco_all_2014.json'),
        os.path.join(root, 'data', 'splits', 'vcoco_{}.ids'.format(split)),
    ]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError('Missing V-COCO evaluation annotation: ' + path)
    spec = importlib.util.spec_from_file_location(
        '_upt_external_vsrl_eval', os.path.join(root, 'vsrl_eval.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator_class = module.VCOCOeval
    exclusions = tuple(sorted(set(
        _action_root(action) for action in exclude_actions)))
    if 'exclude_actions' in inspect.signature(evaluator_class).parameters:
        evaluator = evaluator_class(*paths, exclude_actions=exclusions)
    else:
        # Upstream evaluates every action. Filter its action annotations before
        # it constructs the action/role indices, without editing external files.
        class ProtocolEvaluator(evaluator_class):
            def _init_vcoco(self):
                self.VCOCO = [item for item in self.VCOCO
                              if _action_root(item['action_name']) not in exclusions]
                super()._init_vcoco()

        evaluator = ProtocolEvaluator(*paths)
    if any(_action_root(action) in exclusions for action in evaluator.actions):
        raise RuntimeError('Evaluator did not remove excluded actions: ' + str(exclusions))
    return evaluator


def _action_root(action):
    root = str(action).lower().replace('_', ' ').split()[0]
    return 'point' if root == 'points' else root


def evaluate_vcoco_cache(evaluator, cache_path):
    """Return AP percentages; also supports upstream print-only evaluators."""
    import sys

    report = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, report)):
        returned = evaluator._do_eval(cache_path, ovr_thresh=0.5)
    metrics = dict(returned) if isinstance(returned, dict) else {}
    patterns = {
        'agent_map': r'Average Agent AP\s*=\s*([\d.]+)',
        'scenario1_role_map': r'Average Role \[scenario_1\] AP\s*=\s*([\d.]+)',
        'scenario2_role_map': r'Average Role \[scenario_2\] AP\s*=\s*([\d.]+)',
    }
    for name, pattern in patterns.items():
        if name not in metrics:
            match = re.search(pattern, report.getvalue())
            if not match:
                raise RuntimeError('Evaluator did not report ' + name)
            metrics[name] = float(match.group(1))
        metrics[name] = float(metrics[name])
        if not math.isfinite(metrics[name]):
            raise RuntimeError('Non-finite V-COCO metric: ' + name)
    return {name: metrics[name] for name in patterns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcoco-eval-root', default=None)
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--exclude-actions', nargs='*', default=['point'])
    parser.add_argument('--cache', help='Prediction cache.pkl produced by main.py')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.check_only and not args.cache:
        parser.error('--cache is required unless --check-only is set')
    evaluator = load_evaluator(
        args.vcoco_eval_root, args.split, args.exclude_actions)
    print('V-COCO protocol: split={}, excluded_actions={}'.format(
        args.split, args.exclude_actions))
    if args.check_only:
        print('Evaluator and annotation checks passed.')
    else:
        print(json.dumps(evaluate_vcoco_cache(evaluator, args.cache), indent=2))


if __name__ == '__main__':
    main()
