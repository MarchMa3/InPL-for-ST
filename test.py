import os

# 假设patches文件夹的路径
patches_dir = "/Users/mamarch/Desktop/UWM/ST/InPL-for-ST/hest_data/breast/patches"
# 计算文件数量
files_count = len([f for f in os.listdir(patches_dir) if os.path.isfile(os.path.join(patches_dir, f))])

print(f"文件数量: {files_count}")