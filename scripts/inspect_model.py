from typing import Any

import h5py


def print_hdf5_structure(name: str, obj: Any) -> None:
    print(name)


with h5py.File("src/dive_color_corrector/models/deep_sesr_2x_1d.h5", "r") as f:
    print("HDF5 file structure:")
    f.visititems(print_hdf5_structure)
