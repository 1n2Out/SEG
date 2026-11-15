"""Optional multi-resolution detail fusion; configuration stays PyTorch-free.

Options on top of concat/gated (both off by default, keeping EXP-006 unchanged):
- deep_supervision: auxiliary CE + regional Dice on the 1/2 branch, the backbone's
  own (coarse) logits and the 1/8, 1/16 decoder stages; training only (IDEA-007);
- input_stem: the branch's 1/2 detail, downsampled to 1/4, is added to the patch
  embedding through a zero-initialized projection (IDEA-008).
"""

DEFAULTS = {"mode": "off", "full_channels": 16, "half_channels": 32,
            "deep_supervision": False, "input_stem": False}
MODES = ("off", "concat", "gated")
OPTIONS = ("deep_supervision", "input_stem")


def resolve_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("detail_fusion must be an object")
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown detail_fusion settings: " + ", ".join(sorted(unknown)))
    resolved = {**DEFAULTS, **settings}
    if resolved["mode"] not in MODES:
        raise ValueError("detail_fusion.mode must be off, concat or gated")
    for name in ("full_channels", "half_channels"):
        # The semantic gate compresses to channels // 2, so at least two are needed.
        if type(resolved[name]) is not int or resolved[name] < 2:
            raise ValueError("detail_fusion." + name + " must be an integer of at least 2")
    for name in OPTIONS:
        if type(resolved[name]) is not bool:
            raise ValueError("detail_fusion." + name + " must be true or false")
    if resolved["mode"] == "off" and any(resolved[name] for name in OPTIONS):
        raise ValueError("detail_fusion options require concat or gated mode")
    return resolved
