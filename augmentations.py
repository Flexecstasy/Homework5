import numpy as np
from PIL import ImageFilter, ImageEnhance
from torchvision import transforms
from torchvision.transforms import functional as F

# 1) Стандартные aугментации torchvision
std_augs = {
    'hflip': transforms.RandomHorizontalFlip(p=1.0),
    'crop': transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    'jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    'rot': transforms.RandomRotation(30),
    'gray': transforms.RandomGrayscale(p=1.0),
}
all_augs = transforms.Compose([
    std_augs['hflip'], std_augs['crop'],
    std_augs['jitter'], std_augs['rot'], std_augs['gray']
])

# 2) Кастомные аугментации
class RandomBlur:
    def __init__(self, p=0.5, radius=(0.5,2.0)):
        self.p, self.radius = p, radius
    def __call__(self, img):
        if np.random.rand() < self.p:
            r = np.random.uniform(*self.radius)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img

class RandomPerspectiveTransform:
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p, self.scale = p, distortion_scale
    def __call__(self, img):
        if np.random.rand() < self.p:
            params = transforms.RandomPerspective.get_params(
                img.size, self.scale, self.p)
            return F.perspective(img, *params)
        return img

class RandomBrightnessContrast:
    def __init__(self, p=0.5, brightness=(0.7,1.3), contrast=(0.7,1.3)):
        self.p, self.b, self.c = p, brightness, contrast
    def __call__(self, img):
        if np.random.rand() < self.p:
            img = ImageEnhance.Brightness(img).enhance(
                np.random.uniform(*self.b))
            img = ImageEnhance.Contrast(img).enhance(
                np.random.uniform(*self.c))
        return img

extra_augs = [RandomBlur(), RandomPerspectiveTransform(), RandomBrightnessContrast()]
