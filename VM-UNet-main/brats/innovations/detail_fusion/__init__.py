"""Optional multi-resolution detail fusion; configuration stays PyTorch-free.

Options on top of concat/gated (both off by default, keeping EXP-006 unchanged):
- boundary_weight: weight of an auxiliary WT/TC/ET boundary loss on the 1/1 features;
- context_slices: neighbouring slices per side read by the detail branch only.
"""
import math

DEFAULTS = {"mode": "off", "full_channels": 16, "half_channels": 32,
            "boundary_weight": 0.0, "context_slices": 0}
MODES = ("off", "concat", "gated")
OPTIONS = ("boundary_weight", "context_slices")


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
    weight = resolved["boundary_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("detail_fusion.boundary_weight must be a finite nonnegative number")
    # The approved protocol exception covers one neighbour on each side.
    if type(resolved["context_slices"]) is not int or resolved["context_slices"] not in (0, 1):
        raise ValueError("detail_fusion.context_slices must be 0 or 1")
    if resolved["mode"] == "off" and any(resolved[name] != DEFAULTS[name] for name in OPTIONS):
        raise ValueError("detail_fusion options require concat or gated mode")
    return resolved
