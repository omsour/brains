import os
import numpy as np
import hashlib

def get_file_checksum(file_path):
    """Compute the MD5 checksum of the file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_file(file_path):
    """Check file existence, type, permissions, and attempt to load it."""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if the file is a .npz file
    if not file_path.endswith('.npz'):
        raise ValueError(f"File is not a .npz file: {file_path}")

    # Check file permissions
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")

    # Compute and print file checksum
    checksum = get_file_checksum(file_path)
    print(f"File checksum: {checksum}")

    # Try loading the file
    try:
        with np.load(file_path, allow_pickle=True) as data:
            print("File loaded successfully.")
            # List the content of the .npz file
            for key in data.files:
                print(f"{key}: {data[key].shape}")
    except Exception as e:
        print(f"Error loading .npz file: {e}")

if __name__ == "__main__":
    file_path = "main/mainSamplingData/postprocessed_data.npz"

    # Check the file
    try:
        check_file(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
