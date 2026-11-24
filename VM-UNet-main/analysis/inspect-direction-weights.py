"""读取 EXP-017 val best，在每名验证患者的三张固定轴向切片上测量方向权重。"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from brats.data import load_manifest
from brats.dataset import padding
from brats.engine import predict_volume
from train_brats import build_model, load_checkpoint


def record(scores, masks, name, module, inputs, logits):
    batch, _, height, width = logits.shape
    probabilities = logits.reshape(batch, 4, 8, height, width).float().softmax(1)
    largest = probabilities.max(dim=1).values
    tumor = torch.nn.functional.adaptive_max_pool2d(masks["tumor"], (height, width)).bool()
    values = largest.masked_select(tumor)
    scores[name].append([largest.mean().item(), (largest > 0.5).float().mean().item(),
                         values.sum().item(), values.numel()])


def main():
    directory = SOURCE.parent / "research/experiments/EXP-017"
    summary = json.loads((directory / "summary.json").read_text())
    checkpoint = load_checkpoint(SOURCE / "results" / summary["run_dir"] / "best.pth")
    manifest = load_manifest(SOURCE / "data/brats2020/split_seed42.json")
    if checkpoint["epoch"] != summary["selected_epoch"] or checkpoint["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Unexpected checkpoint epoch or data split")
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval()
    scores = {}
    masks = {}
    for name, module in model.named_modules():
        if name.endswith(".direction_fusion.score"):
            scores[name] = []
            module.register_forward_hook(partial(record, scores, masks, name))
    if len(scores) != 15:
        raise ValueError("Expected 15 direction fusion modules")
    cases = manifest["splits"]["val"]
    for case_id in cases:
        cache = SOURCE / "data/brats2020/cache" / case_id
        volume = np.load(cache / "image.npy", mmap_mode="r")
        indices = [len(volume) * k // 4 for k in (1, 2, 3)]
        labels = np.load(cache / "label.npy", mmap_mode="r")[indices]
        pads = padding(labels.shape[-2:], config["input_size"])
        mask = np.pad(labels > 0, ((0, 0),) + pads)
        masks["tumor"] = torch.from_numpy(mask).to(device).unsqueeze(1).float()
        predict_volume(model, volume[indices], device, config["input_size"], 3, config["amp"])
    blocks = {}
    for name, values in scores.items():
        if len(values) != len(cases) or not np.isfinite(values).all():
            raise ValueError("Incomplete direction-weight observations")
        largest, majority, tumor_sum, tumor_count = np.sum(values, axis=0)
        blocks[name] = {"mean_max_probability": largest / len(cases),
                        "fraction_max_above_half": majority / len(cases),
                        "tumor_cells_mean_max_probability": tumor_sum / tumor_count}
    result = {"epoch": summary["selected_epoch"], "patients": len(cases), "slices_per_patient": 3,
              "scope": "all spatial positions and groups; tumor cells overlap at least one WT voxel",
              "uniform_max_probability": 0.25, "blocks": blocks}
    (directory / "direction-weights.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
