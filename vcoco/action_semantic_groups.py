"""Fixed V-COCO role-action semantic groups for non-gated RGB/DVS branches.

The indices must match ``instances_vcoco_{split}.json['classes']``.  Names are
kept next to indices and validated at startup so annotation changes cannot
silently route an action through the wrong modality branch.
"""

VCOCO_ROLE_ACTIONS = (
    "hold obj", "sit instr", "ride instr", "look obj",
    "hit instr", "hit obj", "eat obj", "eat instr",
    "jump instr", "lay instr", "talk_on_phone instr", "carry obj",
    "throw obj", "catch obj", "cut instr", "cut obj",
    "work_on_computer instr", "ski instr", "surf instr",
    "skateboard instr", "drink instr", "kick obj", "read obj",
    "snowboard instr",
)

# Appearance/pose/object relationship is sufficient; DVS is not injected.
RGB_DOMINANT_ACTIONS = (
    "hold obj", "sit instr", "look obj", "lay instr",
    "talk_on_phone instr", "work_on_computer instr", "read obj",
)

# Both static object evidence and temporal change are useful.
RGB_DVS_JOINT_ACTIONS = (
    "eat obj", "eat instr", "carry obj", "cut instr", "cut obj",
    "drink instr",
)

# Motion direction, burst, or temporal evolution is central to the action.
DVS_STRONG_ACTIONS = (
    "ride instr", "hit instr", "hit obj", "jump instr", "throw obj",
    "catch obj", "ski instr", "surf instr", "skateboard instr",
    "kick obj", "snowboard instr",
)


def vcoco_action_group_indices(classes=None):
    classes = tuple(VCOCO_ROLE_ACTIONS if classes is None else classes)
    if classes != VCOCO_ROLE_ACTIONS:
        raise ValueError(
            "V-COCO action order mismatch. Expected {} but got {}"
            .format(VCOCO_ROLE_ACTIONS, classes)
        )
    name_to_idx = {name: idx for idx, name in enumerate(classes)}
    groups = {
        "rgb": tuple(name_to_idx[name] for name in RGB_DOMINANT_ACTIONS),
        "joint": tuple(name_to_idx[name] for name in RGB_DVS_JOINT_ACTIONS),
        "dynamic": tuple(name_to_idx[name] for name in DVS_STRONG_ACTIONS),
    }
    assigned = sorted(idx for values in groups.values() for idx in values)
    if assigned != list(range(len(classes))):
        raise RuntimeError("V-COCO semantic groups must cover every action exactly once")
    return groups

