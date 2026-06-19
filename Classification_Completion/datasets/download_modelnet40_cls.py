import kagglehub
import shutil
import os

# Download to KaggleHub cache directory
source_path = kagglehub.dataset_download("cuge1995/modelnet40")

# Define the custom target path
target_path = "./modelnet40_ply_hdf5_2048"
os.makedirs(target_path, exist_ok=True)

# Copy all contents from source to target
shutil.copytree(source_path, target_path, dirs_exist_ok=True)

# Remove the original cached source directory
shutil.rmtree(source_path)

print("Dataset downloaded to: ", target_path)
