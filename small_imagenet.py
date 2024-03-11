import os
import shutil
import random

# 定义数据集路径
dataset_dir = '/work/data/imagenet'

# 定义新的小型数据集路径
small_dataset_dir = '/work/data/small_imagenet'

# 定义你想从每个类别中取出的数据的比例
percentage = 0.1

for type in ['train', 'val']:
    image_dir = os.path.join(dataset_dir, type)
    small_image_dir = os.path.join(small_dataset_dir, type)

    # 确保新的小型数据集路径存在
    os.makedirs(small_image_dir, exist_ok=True)

    # 遍历所有的类别
    for category in os.listdir(image_dir):
        category_dir = os.path.join(image_dir, category)
        small_category_dir = os.path.join(small_image_dir, category)

        # 取出所有图片的文件名，并进行随机打乱
        images = os.listdir(category_dir)
        random.shuffle(images)

        # 根据指定的百分比提取文件数
        num_images = int(len(images) * percentage)
        selected_images = images[:num_images]

        # 确保目标目录存在
        os.makedirs(small_category_dir, exist_ok=True)

        # 将被选中的文件从原数据集复制到新的小型数据集
        for image in selected_images:
            src_path = os.path.join(category_dir, image)
            dst_path = os.path.join(small_category_dir, image)

            # 复制文件
            shutil.copy(src_path, dst_path)

print("Finished creating the smaller dataset.")