import itertools
import csv
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from training_dataset_functions import train_epoch, load_HSI, HSI_Dataset
from autoencoder import Autoencoder
import numpy as np
import torch.nn.functional as F
import random
from sklearn.feature_extraction.image import extract_patches_2d
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os

def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in dict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in dict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            dict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[dict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[dict[i]])
            num += 1

    return dict, ASAM / float(num)

def numpy_SAD(y_true, y_pred):
    return np.arccos(np.clip(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)), -1.0, 1.0))

# Define hyperparameter search space
temperature_values = [10] # [0.01, 10, 0.1, 1] 
lambda_recon_values = [0.5, 0.3] # [0.9, 0.7, 0.5, 0.3, 0.1]
patch_size_values = [8, 16, 32, 64]
batch_size_values = [8, 15, 30, 60]
epochs = 100
sigma_kernel_values = [0.01, 0.1, 1, 10]

# List of datasets
datasets = {
    'Samson': {'bands': 156, 'endmembers': 3},
    'Urban4': {'bands': 162, 'endmembers': 4},
    # 'Urban5': {'bands': 162, 'endmembers': 5},
    # 'Urban6': {'bands': 162, 'endmembers': 6},
    # 'JasperRidge': {'bands': 198, 'endmembers': 4},
}

# Best SAD across all datasets
global_best_sad = float('inf')
global_best_hyperparams = None

# Prepare CSV file
csv_filename = "grid_search_results_gen_42.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ["Temperature", "Lambda Recon", "Patch Size", "Batch Size", "Sigma Kernel", "Epochs", "Avg SAD"]
    headers += list(datasets.keys())  # Add dataset names as column headers
    writer.writerow(headers)

# Start grid search
for temperature, lambda_recon, patch_size, batch_size, sigma_kernel in itertools.product(
        temperature_values, lambda_recon_values, patch_size_values, batch_size_values, sigma_kernel_values
    ):
    
    print(f"Testing: Temperature={temperature}, Lambda Recon={lambda_recon}, Patch Size={patch_size}, Batch Size={batch_size}, Sigma Kernel={sigma_kernel}", flush=True)
    
    dataset_sad_values = {}

    for dataset_name, dataset_params in datasets.items():
        
        print(f"Processing dataset: {dataset_name}")
        
        # Clear GPU memory to avoid issues when switching datasets
        torch.cuda.empty_cache()
        
        # Set seed globally at the beginning
        seed_value = 42
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Ensure no randomness in convolutions
        torch.use_deterministic_algorithms(True)  # Force deterministic operations
        np.random.seed(seed_value)
        random.seed(seed_value)

        # DataLoader with seed
        g = torch.Generator()
        g.manual_seed(42)

        os.environ["PYTHONHASHSEED"] = str(seed_value)  # Ensure Python's hash functions are deterministic
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensure deterministic behavior for cuBLAS
        
        params = {
            'n_bands': dataset_params['bands'],
            'e_filters': 48,
            'e_size': 3,
            'd_filters': dataset_params['bands'],
            'd_size': 13,
            'num_endmembers': dataset_params['endmembers'],
            'scale': 3,
            'lr': 0.003,
            'weight_decay': 1e-6,
            'alpha_range': (0.8, 1.2), 
            'num_patches': 250
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        data, GT, S_GT = load_HSI(f'./Datasets/{dataset_name}.mat')

        # Prepare dataset
        hsi_dataset = HSI_Dataset(data, patch_size, params['num_patches'], alpha_range=params['alpha_range'], random_state=42)
        data_loader = DataLoader(
                    hsi_dataset,
                    batch_size=batch_size,
                    shuffle=True,  # Keep shuffle but ensure consistent results
                    collate_fn=lambda x: (
                        torch.stack([item[0] for item in x]),  # Patches
                        torch.tensor([item[1] for item in x])  # Alphas
                    ),
                    num_workers=0,  # Setting workers to 0 ensures full reproducibility (multi-threading can introduce randomness)
                    worker_init_fn=lambda worker_id: np.random.seed(seed_value + worker_id),  # Ensure workers get the same seed
                    generator=g
                )
        
        model = Autoencoder({**params, 'patch_size': patch_size}).to(device)
        optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # Extract labels for contrastive learning
        patches = extract_patches_2d(S_GT, (patch_size, patch_size), max_patches=params['num_patches'], random_state=42)
        labels = patches.mean(axis=(1, 2))
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        # Train model
        for epoch in range(epochs):
            total_loss, contrastive_loss, recon_loss = train_epoch(
                model, data_loader, optimizer, labels, device,
                temperature=temperature, lambda_recon=lambda_recon,
                loss_type='generalized', kernel='rbf', sigma_kernel=sigma_kernel
            )
            print(f"Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss:.4f}, Contrastive Loss: {contrastive_loss:.4f}, Recon Loss: {recon_loss:.4f}", flush=True)

        # Extract non-negative endmembers from the decoder
        endmembers = F.relu(model.decoder.output_layer.weight_raw).detach().cpu().numpy()

        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        else:
            endmembers = np.squeeze(endmembers)

        predicted_endmembers = endmembers.T
        true_endmembers = GT

        # Normalize endmembers
        for m in range(true_endmembers.shape[0]):
            predicted_endmembers[m, :] = predicted_endmembers[m, :] / predicted_endmembers[m, :].max()
            true_endmembers[m, :] = true_endmembers[m, :] / true_endmembers[m, :].max()

        # Calculate SAD values
        order_dict, mean_sad = order_endmembers(true_endmembers, predicted_endmembers)
        reordered_predicted_endmembers = predicted_endmembers[[order_dict[k] for k in sorted(order_dict.keys())]]

        # Store SAD for this dataset
        dataset_sad_values[dataset_name] = mean_sad
        
        run_sad_values = [numpy_SAD(reordered_predicted_endmembers[j, :], true_endmembers[j, :]) for j in range(true_endmembers.shape[0])]
        
        print(f"Mean SAD: {mean_sad:.4f}", flush=True)
        print(f"Run SAD: {run_sad_values}", flush=True)
        
        # Cleaning all variables to avoid memory leaks
        del model, optimizer, data_loader, hsi_dataset, data, GT, S_GT, patches, labels
        torch.cuda.empty_cache()

    # Compute the average SAD across all datasets
    avg_sad = np.mean(list(dataset_sad_values.values()))
    
    print(f"Average SAD: {avg_sad:.4f}", flush=True)

    # Save results to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temperature, lambda_recon, patch_size, batch_size, sigma_kernel, epochs, avg_sad] + list(dataset_sad_values.values()))
        

    # Update best hyperparameters if the average SAD is lower
    if avg_sad < global_best_sad:
        global_best_sad = avg_sad
        global_best_hyperparams = {
            'temperature': temperature,
            'lambda_recon': lambda_recon,
            'patch_size': patch_size,
            'batch_size': batch_size,
            'sigma_kernel': sigma_kernel,
            'epochs': epochs
        }

# Print the best configuration
print("\nBest Hyperparameters Found:")
print(global_best_hyperparams)
print(f"Best Average SAD: {global_best_sad:.4f}")
print(f"Results saved to {csv_filename}")