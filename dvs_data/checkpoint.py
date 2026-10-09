"""Explicit weight migration from the retired global-mean DVS branch."""


LEGACY_KEYS = {
    'interaction_head.dvs_norm.weight', 'interaction_head.dvs_norm.bias',
    'interaction_head.dvs_adapter.weight', 'interaction_head.dvs_adapter.bias',
}
QUERY_PREFIX = 'interaction_head.dvs_query.'


def is_legacy_dvs_state(state):
    return bool(LEGACY_KEYS.intersection(state))


def load_hoi_weights(model, state, init_legacy_dvs=False):
    if not is_legacy_dvs_state(state):
        if init_legacy_dvs:
            raise ValueError('--init-legacy-dvs requires a legacy mean/residual checkpoint')
        model.load_state_dict(state)
        return []
    if not init_legacy_dvs:
        raise ValueError(
            'This checkpoint uses the retired temporal-mean DVS branch. '
            'For a new query experiment use --resume PATH --init-legacy-dvs; '
            'do not use --resume-training or --eval with legacy DVS weights.')
    expected = model.state_dict()
    new_keys = {key for key in expected if key.startswith(QUERY_PREFIX)}
    if not new_keys:
        raise ValueError('Legacy DVS initialization requires --use-dvs with the query model')
    shared = {key: value for key, value in state.items() if key not in LEGACY_KEYS}
    missing = set(expected) - set(shared)
    unexpected = set(shared) - set(expected)
    if missing != new_keys or unexpected:
        raise ValueError('Legacy checkpoint has incompatible shared layers: missing={} '
                         'unexpected={}'.format(sorted(missing - new_keys), sorted(unexpected)))
    model.load_state_dict(shared, strict=False)
    return sorted(LEGACY_KEYS.intersection(state))
