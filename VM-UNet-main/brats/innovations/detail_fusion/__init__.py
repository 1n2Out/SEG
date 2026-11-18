"""Optional multi-resolution detail fusion; configuration stays PyTorch-free."""

DEFAULTS = {"mode": "off", "full_channels": 16, "half_channels": 32}
MODES = ("off", "concat", "gated")
# Retired options (IDEA-007 deep supervision, IDEA-008 input stem). Checkpoints from
# that code, e.g. the concat ablation EXP-014, record them as false; those still load.
RETIRED_OPTIONS = ("deep_supervision", "input_stem")


def resolve_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("detail_fusion must be an object")
    settings = dict(settings)
    for name in RETIRED_OPTIONS:
        if name in settings:
            if settings.pop(name) is not False:
                raise ValueError("detail_fusion." + name + " has been retired (IDEA-007/008); restore "
                                 "its source from git commit 6785299 to evaluate or resume old runs")
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
    return resolved
