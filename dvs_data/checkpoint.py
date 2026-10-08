"""Strict DVS weight loading and explicit initialization of new query branches."""


LEGACY_KEYS = {
    'interaction_head.dvs_norm.weight', 'interaction_head.dvs_norm.bias',
    'interaction_head.dvs_adapter.weight', 'interaction_head.dvs_adapter.bias',
}
QUERY_PREFIX = 'interaction_head.dvs_query.'
EXTENSION_PREFIXES = tuple(QUERY_PREFIX + name + '.' for name in
                           ('change_norm', 'change_proj', 'token_type', 'interval_proj')) + (
                               'interaction_head.dvs_relation_query.',)


def is_legacy_dvs_state(state):
    return bool(LEGACY_KEYS.intersection(state))


def query_extension_keys(state):
    return {key for key in state if key.startswith(EXTENSION_PREFIXES)}


def validate_query_layout(state, adjacent_changes, precomp_residual, init_query_baseline=False):
    """The optional branches have distinct state keys, including token types."""
    if is_legacy_dvs_state(state):
        if init_query_baseline:
            raise ValueError('--init-query-baseline requires a temporal query checkpoint')
        return  # Legacy initialization is checked separately.
    saved_changes = any(key.startswith(QUERY_PREFIX + 'change_proj.') for key in state)
    saved_relation = any(key.startswith('interaction_head.dvs_relation_query.') for key in state)
    if init_query_baseline:
        if (saved_changes or saved_relation or not (adjacent_changes or precomp_residual)
                or not any(key.startswith(QUERY_PREFIX) for key in state)):
            raise ValueError('--init-query-baseline needs a baseline query checkpoint and a new DVS branch')
    elif (saved_changes, saved_relation) != (adjacent_changes, precomp_residual):
        raise ValueError('DVS checkpoint branch mismatch: saved changes/relation={}/{}, requested={}/{}. '
                         'Match the branch flags, or initialize a NEW experiment with '
                         '--init-query-baseline.'.format(saved_changes, saved_relation,
                                                        adjacent_changes, precomp_residual))


def load_hoi_weights(model, state, init_legacy_dvs=False, init_query_baseline=False):
    if init_legacy_dvs and init_query_baseline:
        raise ValueError('Choose only one checkpoint initialization mode')
    expected = model.state_dict()
    validate_query_layout(
        state, any(key.startswith(QUERY_PREFIX + 'change_proj.') for key in expected),
        any(key.startswith('interaction_head.dvs_relation_query.') for key in expected),
        init_query_baseline)
    if init_query_baseline:
        new_keys = query_extension_keys(expected)
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        if missing != new_keys or unexpected:
            raise ValueError('Query baseline has incompatible shared layers: missing={} unexpected={}'.format(
                sorted(missing - new_keys), sorted(unexpected)))
        model.load_state_dict(state, strict=False)
        return sorted(new_keys)
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
    new_keys = {key for key in expected if key.startswith(QUERY_PREFIX)} | query_extension_keys(expected)
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
