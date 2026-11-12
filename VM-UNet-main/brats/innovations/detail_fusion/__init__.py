"""Optional multi-resolution detail fusion; configuration stays PyTorch-free."""

DEFAULTS = {"mode": "off", "full_channels": 16, "half_channels": 32}
MODES = ("off", "concat", "gated")


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
    return resolved
