"""Memory-mapped axial slices. All slices, including tumor-negative ones, remain."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import check_cache


def padding(shape, input_size):
    if input_size % 32 != 0 or any(s > input_size for s in shape):
        raise ValueError(f"Input size {input_size} must be divisible by 32 and cover {shape}")
    return tuple(((input_size - s) // 2, input_size - s - (input_size - s) // 2) for s in shape)


def left_right_axis(metadata):
    from nibabel.orientations import aff2axcodes

    axes = aff2axcodes(np.asarray(metadata["affine"]))
    for axis in (0, 1):
        if axes[axis] in ("L", "R"):
            return axis
    raise ValueError(f"Left/right axis must lie in the slice plane: {metadata['case_id']} {axes}")


def pack_contralateral(image, axis):
    """Append signed differences before augmentation or padding, for (..., C, X, Y)."""
    if axis not in (0, 1):
        raise ValueError("Expected a left/right axis of 0 or 1")
    difference = image - np.flip(image, axis=axis - 2)
    return np.concatenate((image, difference), axis=-3)


class BraTSSlices(Dataset):
    def __init__(self, manifest, cache_dir, split="train", input_size=256, augment=True,
                 contralateral=False):
        if split not in ("train", "val", "test"):
            raise ValueError("Supervised slices require an internal labeled split")
        self.cache_dir, self.input_size = Path(cache_dir), input_size
        self.augment = augment and split == "train"
        self.index = []
        self.maps = {}
        self.lr_axes = {}
        for case_id in manifest["splits"][split]:
            metadata = check_cache(cache_dir, manifest["cases"][case_id])
            if contralateral:
                self.lr_axes[case_id] = left_right_axis(metadata)
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
        image = np.array(images[z], dtype=np.float32)
        if case_id in self.lr_axes:
            image = pack_contralateral(image, self.lr_axes[case_id])
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
