import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import CustomImageDataset

DATA_TRAIN = 'data/train'
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def class_counts():
    ds = CustomImageDataset(DATA_TRAIN)
    counts = {cls: 0 for cls in ds.get_class_names()}
    for _, label in ds:
        counts[ds.get_class_names()[label]] += 1
    return counts

def image_size_stats():
    sizes = []
    for root, _, files in os.walk(DATA_TRAIN):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                w,h = Image.open(os.path.join(root,f)).size
                sizes.append((w,h))
    ws, hs = zip(*sizes)
    return {
        'min': [int(min(ws)), int(min(hs))],
        'max': [int(max(ws)), int(max(hs))],
        'mean': [int(np.mean(ws)), int(np.mean(hs))],
        'total_images': len(sizes)
    }, ws, hs

def save_stats():
    counts = class_counts()
    stats, ws, hs = image_size_stats()
    payload = {'by_class': counts, **stats}
    with open(os.path.join(RESULTS_DIR, 'stats.json'), 'w') as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    return counts, ws, hs

def plot_and_save_hist(data, name, xlabel):
    plt.figure()
    plt.hist(data, bins=20)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name.lower().replace(' ','_')}.png"))
    plt.close()

def plot_class_counts(counts):
    plt.figure(figsize=(6,4))
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Images per Class')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "images_per_class.png"))
    plt.close()

if __name__ == '__main__':
    counts, ws, hs = save_stats()
    plot_and_save_hist(ws, 'Width Distribution', 'Width (px)')
    plot_and_save_hist(hs, 'Height Distribution', 'Height (px)')
    plot_class_counts(counts)
    print(" analysis.py: stats.json and plots saved.")
