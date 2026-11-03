from .vmamba import VSSM
from collections.abc import Mapping
import hashlib
from pathlib import Path
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        if x.size()[1] == 1 and self.input_channels == 3:
            x = x.repeat(1,3,1,1)
        if x.size()[1] != self.input_channels:
            raise ValueError(f'Expected {self.input_channels} input channels, got {x.size()[1]}')
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is None:
            return {"loaded": [], "adapted": [], "skipped": []}
        path = Path(self.load_ckpt_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Pretrained checkpoint not found: {path}')
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:  # Older PyTorch releases lack weights_only.
            checkpoint = torch.load(path, map_location='cpu')
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Expected a VMamba checkpoint containing model_ema or model weights')
        checkpoint_key = next((key for key in ('model_ema', 'model')
                               if isinstance(checkpoint.get(key), Mapping) and checkpoint[key]), None)
        if checkpoint_key is None:
            raise ValueError('Expected a VMamba checkpoint containing model_ema or model weights')
        pretrained = {key: value for key, value in checkpoint[checkpoint_key].items()
                      if torch.is_tensor(value)}
        candidates = dict(pretrained)
        for key, value in pretrained.items():
            for stage in range(4):
                prefix = f'layers.{stage}.'
                if key.startswith(prefix):
                    candidates[key.replace(prefix, f'layers_up.{3-stage}.', 1)] = value
                    break
        target = self.vmunet.state_dict()
        selected = {}
        report = {"loaded": [], "adapted": [], "skipped": [], "unexpected": []}
        for key, value in candidates.items():
            if key not in target:
                report['unexpected'].append(key)
                continue
            # RGB -> MRI channels: average RGB kernels, repeat, preserve summed
            # response to identical channels. Other mismatches are never forced.
            if (key == 'patch_embed.proj.weight' and value.ndim == 4 and
                    target[key].ndim == 4 and value.shape[1] == 3 and
                    value.shape[1] != target[key].shape[1] and
                    value.shape[0] == target[key].shape[0] and
                    value.shape[2:] == target[key].shape[2:]):
                channels = target[key].shape[1]
                value = value.mean(dim=1, keepdim=True).repeat(1, channels, 1, 1) * (3.0 / channels)
                report['adapted'].append(key)
            if value.shape == target[key].shape:
                selected[key] = value
                report['loaded'].append(key)
            else:
                report['skipped'].append(key)
        if not selected:
            raise ValueError('No compatible VMamba pretrained parameters found')
        self.vmunet.load_state_dict(selected, strict=False)
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                checksum.update(chunk)
        report.update(source_path=str(path), checkpoint_key=checkpoint_key,
                      source_sha256=checksum.hexdigest(),
                      loaded_numel=sum(value.numel() for value in selected.values()),
                      target_numel=sum(value.numel() for value in target.values()),
                      missing=sorted(set(target) - set(selected)))
        report['loaded_fraction'] = report['loaded_numel'] / report['target_numel']
        print(f"Pretrained [{checkpoint_key}]: {path}; loaded={len(selected)}, "
              f"adapted={report['adapted']}, shape-skipped={len(report['skipped'])}, "
              f"missing={len(report['missing'])}, target coverage={report['loaded_fraction']:.1%}", flush=True)
        return report
