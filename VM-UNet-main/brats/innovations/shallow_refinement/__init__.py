"""Optional shallow skip refinement; configuration stays PyTorch-free."""

DEFAULTS = {"mode": "off", "hidden_channels": 24}
MODES = ("off", "simple", "gated")


def resolve_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("shallow_refinement must be an object")
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown shallow_refinement settings: " + ", ".join(sorted(unknown)))
    resolved = {**DEFAULTS, **settings}
    if resolved["mode"] not in MODES:
        raise ValueError("shallow_refinement.mode must be off, simple or gated")
    if type(resolved["hidden_channels"]) is not int or resolved["hidden_channels"] < 1:
        raise ValueError("shallow_refinement.hidden_channels must be a positive integer")
    return resolved
