import os
import shutil
import random


def split_dataset(train_dir='data/train', val_dir='data/val', split_ratio=0.2, seed=42):
    random.seed(seed)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_train_path):
            continue

        class_val_path = os.path.join(val_dir, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        images = [f for f in os.listdir(class_train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        val_count = int(len(images) * split_ratio)

        val_images = images[:val_count]
        print(f"{class_name}: {val_count} -> validation, {len(images) - val_count} -> train")

        for img_name in val_images:
            src_path = os.path.join(class_train_path, img_name)
            dst_path = os.path.join(class_val_path, img_name)
            shutil.move(src_path, dst_path)




if __name__ == '__main__':
    split_dataset()
