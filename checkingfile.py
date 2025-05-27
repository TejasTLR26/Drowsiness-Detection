import h5py
file_path = "models/model.h5"
try:
    with h5py.File(file_path, "r") as f:
        print("HDF5 file opened successfully.")
except OSError:
    print("HDF5 file is corrupted.")