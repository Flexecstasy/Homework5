import os
import csv
import time
import psutil
import tracemalloc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CustomImageDataset
from augmentations import all_augs

DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_experiment(sizes, n_samples=100):
    results = []
    for sz in sizes:
        tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            all_augs,
            transforms.ToTensor()
        ])
        ds = CustomImageDataset(DATA_DIR, transform=tf)
        loader = DataLoader(ds, batch_size=1, shuffle=True)

        start_rss = psutil.Process().memory_info().rss
        tracemalloc.start()
        t0 = time.time()
        for i, (x, _) in enumerate(loader):
            if i >= n_samples:
                break
        t1 = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_rss = psutil.Process().memory_info().rss

        results.append({
            'size': sz,
            'time_s': round(t1 - t0, 4),
            'peak_mem_mb': round(peak / 1e6, 2),
            'rss_diff_mb': round((end_rss - start_rss) / 1e6, 2)
        })
    return results

def save_results_csv(results, filename=os.path.join(RESULTS_DIR, 'experiment.csv')):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results):
    sizes = [r['size'] for r in results]
    times = [r['time_s'] for r in results]
    mems = [r['peak_mem_mb'] for r in results]
    rss = [r['rss_diff_mb'] for r in results]

    plt.figure(); plt.plot(sizes, times, marker='o'); plt.title('Time vs Size'); plt.xlabel('Size (px)'); plt.ylabel('Time (s)')
    plt.grid(); plt.savefig(os.path.join(PLOTS_DIR, 'experiment_time.png')); plt.close()

    plt.figure(); plt.plot(sizes, mems, marker='o'); plt.title('Peak Mem vs Size'); plt.xlabel('Size (px)'); plt.ylabel('Peak Mem (MB)')
    plt.grid(); plt.savefig(os.path.join(PLOTS_DIR, 'experiment_peak_mem.png')); plt.close()

    plt.figure(); plt.plot(sizes, rss, marker='o'); plt.title('RSS Diff vs Size'); plt.xlabel('Size (px)'); plt.ylabel('RSS Diff (MB)')
    plt.grid(); plt.savefig(os.path.join(PLOTS_DIR, 'experiment_rss_diff.png')); plt.close()

if __name__ == '__main__':
    sizes = [64, 128, 224, 512]
    results = run_experiment(sizes)
    save_results_csv(results)
    plot_results(results)
    print(f" experiment.py: results.csv and plots saved ({len(results)} entries).")
