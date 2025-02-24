import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from sklearn.feature_extraction.image import extract_patches_2d

# Load dataset function
def load_HSI(path):
    data = sio.loadmat(path)
    Y = np.asarray(data['Y'], dtype=np.float32)
    GT = np.asarray(data['GT'], dtype=np.float32)
    S_GT = np.asarray(data['S_GT'], dtype=np.float32) if 'S_GT' in data else None
    
    if Y.shape[0] < Y.shape[1]:
        Y = Y.transpose()
        
    Y = Y / np.max(Y.flatten())
    n_bands = Y.shape[1]
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    Y = np.reshape(Y, (n_cols, n_rows, n_bands))
    
    # Reshape S_GT if necessary
    if S_GT is not None:
        if S_GT.shape[0] == n_cols * n_rows:  # (pixels, num_endmembers)
            S_GT = S_GT.reshape(n_cols, n_rows, -1)
        elif S_GT.shape[1] == n_cols * n_rows:  # (num_endmembers, pixels)
            S_GT = S_GT.T.reshape(n_cols, n_rows, -1)
    
    return Y, GT, S_GT

# Extract a single patch and its abundances
def extract_patch(hsi, s_gt, patch_size=8):
    H, W, C = hsi.shape
    _, _, E = s_gt.shape if s_gt is not None else (None, None, None)  # E: Number of endmembers
    
    x = random.randint(0, H - patch_size)
    y = random.randint(0, W - patch_size)

    patch = hsi[x:x+patch_size, y:y+patch_size, :]
    s_gt_patch = s_gt[x:x+patch_size, y:y+patch_size, :] if s_gt is not None else None

    return patch, s_gt_patch, (x, y)

# Visualize multiple patches and their endmember composition
def visualize_patches(patches, s_gt_patches, dataset_name):
    num_patches = len(patches)
    
    fig, axes = plt.subplots(num_patches, 2, figsize=(12, 3 * num_patches))

    for i in range(num_patches):
        patch = patches[i]
        s_gt_patch = s_gt_patches[i]
        avg_abundances = np.mean(s_gt_patch, axis=(0, 1)) if s_gt_patch is not None else None

        # Display the patch (simulated grayscale representation)
        rgb_patch = np.mean(patch, axis=2)
        axes[i, 0].imshow(rgb_patch, cmap='gray')
        axes[i, 0].set_title(f"{dataset_name} - Patch {i+1}")
        axes[i, 0].axis("off")

        # Endmember abundance composition
        if avg_abundances is not None:
            axes[i, 1].bar(range(len(avg_abundances)), avg_abundances, color='b', alpha=0.6)
            axes[i, 1].set_xlabel("Endmember Index")
            axes[i, 1].set_ylabel("Average Abundance")
            axes[i, 1].set_title(f"Patch {i+1} - Endmember Abundances")

    plt.tight_layout()
    plt.show()

# Process all datasets
datasets = {
    "Samson": "./Datasets/Samson.mat",
    "Urban4": "./Datasets/Urban4.mat",
    "Urban5": "./Datasets/Urban5.mat",
    "Urban6": "./Datasets/Urban6.mat",
    "JasperRidge": "./Datasets/JasperRidge.mat"
}

num_patches_per_dataset = 6  # Set the number of patches to extract per dataset

for dataset_name, dataset_path in datasets.items():
    print(f"Processing {dataset_name} dataset...")
    
    try:
        data, GT, S_GT = load_HSI(dataset_path)

        # Extract multiple patches
        patches = []
        s_gt_patches = []
        for _ in range(num_patches_per_dataset):
            patch, s_gt_patch, _ = extract_patch(data, S_GT)
            patches.append(patch)
            s_gt_patches.append(s_gt_patch)

        # Visualize extracted patches
        visualize_patches(patches, s_gt_patches, dataset_name)
    
    except ValueError as e:
        print(f"Skipping {dataset_name} due to error: {e}")