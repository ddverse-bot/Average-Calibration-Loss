import os
import h5py
import numpy as np
from monai.transforms import MapTransform

class LoadH5d(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            with h5py.File(filepath, 'r') as f:
                loaded_array = f[key][()]
                if loaded_array.ndim == 3:
                    d[key] = np.expand_dims(loaded_array, axis=0)
                elif loaded_array.ndim == 4:
                    d[key] = loaded_array
                else:
                    raise ValueError(f"Unexpected array dimension for key {key}: {loaded_array.ndim}. Expected 3 or 4.")
        return d
