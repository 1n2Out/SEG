"""Memory-mapped axial slices. All slices, including tumor-negative ones, remain.

By default a sample is one axial slice. A method may also read k neighbouring
slices per side (context_slices); they are stacked after the current slice.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import check_cache


def padding(shape, input_size):
    if input_size % 32 != 0 or any(s > input_size for s in shape):
        raise ValueError(f"Input size {input_size} must be divisible by 32 and cover {shape}")
    return tuple(((input_size - s) // 2, input_size - s - (input_size - s) // 2) for s in shape)


def context_offsets(context_slices):
    """Channel-block order of the stacked slices: current, -1, +1, -2, +2, ..."""
    if type(context_slices) is not int or context_slices < 0:
        raise ValueError("context_slices must be a nonnegative integer")
    offsets = [0]
    for distance in range(1, context_slices + 1):
        offsets += [-distance, distance]
    return offsets


def stack_slices(volume, indices, context_slices):
    """Z,C,X,Y volume -> N,C*(2k+1),X,Y float32. Neighbours outside the volume repeat its edge slice."""
    depth = volume.shape[0]
    offsets = context_offsets(context_slices)
    return np.stack([np.concatenate([np.asarray(volume[min(max(z + offset, 0), depth - 1)], dtype=np.float32)
                                     for offset in offsets]) for z in indices])


class BraTSSlices(Dataset):
    def __init__(self, manifest, cache_dir, split="train", input_size=256, augment=True, context_slices=0):
        if split not in ("train", "val", "test"):
            raise ValueError("Supervised slices require an internal labeled split")
        context_offsets(context_slices)
        self.cache_dir, self.input_size = Path(cache_dir), input_size
        self.context_slices = context_slices
        self.augment = augment and split == "train"
        self.index = []
        self.maps = {}
        for case_id in manifest["splits"][split]:
            metadata = check_cache(cache_dir, manifest["cases"][case_id])
            x, y, z = metadata["shape_xyz"]
            padding((x, y), input_size)
            self.index.extend((case_id, k) for k in range(z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        case_id, z = self.index[index]
        # Keep only one subject open per worker, avoiding thousands of file handles.
        if case_id not in self.maps:
            self.maps.clear()
            directory = self.cache_dir / case_id
            self.maps[case_id] = (np.load(directory / "image.npy", mmap_mode="r"),
                                  np.load(directory / "label.npy", mmap_mode="r"))
        images, labels = self.maps[case_id]
        image = (np.array(images[z], dtype=np.float32) if not self.context_slices
                 else stack_slices(images, [z], self.context_slices)[0])
        label = np.array(labels[z], dtype=np.int64)
        label[label == 4] = 3
        if self.augment:
            k = int(np.random.randint(4))
            image, label = np.rot90(image, k, axes=(1, 2)), np.rot90(label, k)
            for axis in (0, 1):
                if np.random.random() < 0.5:
                    image, label = np.flip(image, axis + 1), np.flip(label, axis)
        pads = padding(label.shape, self.input_size)
        image = np.pad(image, ((0, 0),) + pads)
        label = np.pad(label, pads)
        return {"image": torch.from_numpy(np.ascontiguousarray(image)),
                "label": torch.from_numpy(np.ascontiguousarray(label)),
                "case_name": case_id, "slice_index": z}
