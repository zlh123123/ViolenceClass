# Function: 将数据集中的图片按照文件名的第一个字符进行分类，0开头的文件移动到safe文件夹，1开头的文件移动到violence文件夹

import os
import shutil

# 源文件夹路径
source_folder = "./val_ori"

# 目标文件夹路径
safe_folder = "./val/safe"
violence_folder = "./val/violence"

# 创建目标文件夹
if not os.path.exists(safe_folder):
    os.makedirs(safe_folder)
if not os.path.exists(violence_folder):
    os.makedirs(violence_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 判断文件名的第一个字符
    if filename[0] == "0":
        # 将文件移动到safe文件夹
        shutil.move(os.path.join(source_folder, filename), safe_folder)
    elif filename[0] == "1":
        # 将文件移动到violence文件夹
        shutil.move(os.path.join(source_folder, filename), violence_folder)

print("分类完成!")
