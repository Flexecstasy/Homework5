import os
from augmentations import std_augs, all_augs
from PIL import Image

class AugmentationPipeline:
    """
    Позволяет динамически настраивать и сохранять примеры аугментаций.
    """
    def __init__(self):
        self.augs = {}

    def add_augmentation(self, name, aug):
        self.augs[name] = aug

    def remove_augmentation(self, name):
        self.augs.pop(name, None)

    def apply(self, img):
        for aug in self.augs.values():
            img = aug(img)
        return img

    def get_augmentations(self):
        return list(self.augs.keys())

    def save_samples(self, images, out_dir='plots/pipeline_samples'):
        """
        Сохраняет для каждого изображения последовательность:
        original + все по-отдельности + применение всех вместе.
        """
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(images):
            # оригинал
            img.save(f"{out_dir}/img{idx}_orig.jpg")
            # по-отдельности
            for name, aug in self.augs.items():
                aug(img).save(f"{out_dir}/img{idx}_{name}.jpg")
            # вместе
            composed = all_augs(img)
            composed.save(f"{out_dir}/img{idx}_all.jpg")

# Фабрика конфигураций
def build_pipelines():
    configs = {
        'light': ['hflip','crop'],
        'medium': ['hflip','crop','jitter'],
        'heavy': ['hflip','crop','jitter','rot','gray']
    }
    pipelines = {}
    for lvl, names in configs.items():
        p = AugmentationPipeline()
        for n in names:
            p.add_augmentation(n, std_augs[n])
        pipelines[lvl] = p
    return pipelines

if __name__ == '__main__':
    from datasets import CustomImageDataset
    ds = CustomImageDataset('data/train', transform=None)
    samples = [ds[i][0] for i in range(min(3, len(ds)))]
    pipes = build_pipelines()
    for name, p in pipes.items():
        p.save_samples(samples, out_dir=f"plots/{name}_samples")
    print(" pipeline.py: sample images saved.")
