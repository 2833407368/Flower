import os
from PIL import Image

def check_corrupted_images(image_dir):
    corrupted = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img.verify()  # 验证图像完整性
        except (IOError, OSError) as e:
            corrupted.append(img_path)
            print(f"损坏的图像: {img_path} - {e}")
    return corrupted

# 使用方法
TRAIN_IMAGE_DIR = "dataset/trainSet"
corrupted_images = check_corrupted_images(TRAIN_IMAGE_DIR)

# 可选：删除损坏的图像
for img_path in corrupted_images:
    os.remove(img_path)
    print(f"已删除损坏图像: {img_path}")

