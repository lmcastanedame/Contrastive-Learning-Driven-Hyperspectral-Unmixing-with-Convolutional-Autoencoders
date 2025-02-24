import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from training_dataset_functions import train_epoch, load_HSI, HSI_Dataset
from autoencoder import Autoencoder
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import seaborn as sns
import random
import os

def order_endmembers(endmembers, endmembersGT):
    """
    Orders the endmembers based on the similarity between the given endmembers and the ground truth endmembers.
    Args:
        endmembers (numpy.ndarray): Array of shape (num_endmembers, num_features) representing the given endmembers.
        endmembersGT (numpy.ndarray): Array of shape (num_endmembers, num_features) representing the ground truth endmembers.
    Returns:
        dict: A dictionary where the keys represent the indices of the given endmembers and the values represent the indices of the corresponding ground truth endmembers.
        float: The average Spectral Angle Distance (SAD) between the given endmembers and their corresponding ground truth endmembers.
    Raises:
        None
    Notes:
        - The similarity between two endmembers is calculated using the Spectral Angle Distance (SAD) metric.
        - The function orders the endmembers by finding the minimum SAD between each given endmember and each ground truth endmember.
        - The function also calculates the average SAD between the given endmembers and their corresponding ground truth endmembers.
    """
    
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
    """
    Calculates the SAD (Sum of Absolute Differences) between two vectors using numpy.
    Parameters:
        y_true (numpy.ndarray): The true vector.
        y_pred (numpy.ndarray): The predicted vector.
    Returns:
        float: The SAD between the two vectors.
    """
    
    return np.arccos(np.clip(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)), -1.0, 1.0))

def plot_endmembers(true_endmembers, predicted_endmembers):
    """
    Plots the true and predicted endmembers in separate subplots.
    Each subplot corresponds to a single endmember.
    """
    # Normalize each set independently
    true_endmembers = true_endmembers / np.max(true_endmembers, axis=1, keepdims=True)
    predicted_endmembers = predicted_endmembers / np.max(predicted_endmembers, axis=1, keepdims=True)
    
    num_endmembers = true_endmembers.shape[0]

    # Create subplots
    fig, axes = plt.subplots(num_endmembers, 1, figsize=(8, 2 * num_endmembers), sharex=True)

    # Generate distinct colors for each endmember
    colors = sns.color_palette("husl", num_endmembers)  

    for i in range(num_endmembers):
        ax = axes[i] if num_endmembers > 1 else axes  # Handle single subplot case
        ax.plot(true_endmembers[i, :], label=f'True {i+1}', linestyle='-', color=colors[i])
        ax.plot(predicted_endmembers[i, :], label=f'Predicted {i+1}', linestyle='--', color=colors[i])
        ax.legend()
        ax.set_ylabel("Reflectance")
        ax.set_title(f"Endmember {i+1}")

    plt.xlabel("Bands")
    plt.tight_layout()
    plt.show()
    
# Function to plot losses
def plot_losses(contrastive_losses, recon_losses, total_losses, dataset_name):
    """
    Plots the contrastive loss and reconstruction loss against the number of epochs.
    Args:
        contrastive_losses (list): List of contrastive losses for each epoch.
        recon_losses (list): List of reconstruction losses for each epoch.
        total_losses (list): List of total losses for each epoch.
        dataset_name (str): Name of the dataset.
    Returns:
        None
    """
    
    epochs = range(1, len(contrastive_losses) + 1)

    plt.figure(figsize=(10, 4))

    # Contrastive Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, contrastive_losses, label="Contrastive Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Contrastive Loss vs. Epochs ({dataset_name})")
    plt.legend()

    # Reconstruction Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, recon_losses, label="Reconstruction Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Reconstruction Loss vs. Epochs ({dataset_name})")
    plt.legend()

    plt.tight_layout()
    plt.show()

# List of datasets
datasets = {
    'Samson': {'bands': 156, 'endmembers': 3},
    'Urban4': {'bands': 162, 'endmembers': 4},
    # 'Urban5': {'bands': 162, 'endmembers': 5},
    # 'Urban6': {'bands': 162, 'endmembers': 6},
    # 'Cuprite_fixed': {'bands': 188, 'endmembers': 12},
    'JasperRidge': {'bands': 198, 'endmembers': 4},
}

# Set best hyperparameters
best_hyperparams = {
    'temperature': 0.1,
    'lambda_recon': 0.5,
    'patch_size': 32,
    'batch_size': 8,
    'epochs': 100
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # Fixed parameters
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
    hsi_dataset = HSI_Dataset(data, best_hyperparams['patch_size'], params['num_patches'], alpha_range=params['alpha_range'], random_state=42)
    data_loader = DataLoader(
                    hsi_dataset,
                    batch_size=best_hyperparams['batch_size'],
                    shuffle=True,  # Keep shuffle but ensure consistent results
                    collate_fn=lambda x: (
                        torch.stack([item[0] for item in x]),  # Patches
                        torch.tensor([item[1] for item in x])  # Alphas
                    ),
                    num_workers=0,  # Setting workers to 0 ensures full reproducibility (multi-threading can introduce randomness)
                    worker_init_fn=lambda worker_id: np.random.seed(seed_value + worker_id),  # Ensure workers get the same seed
                    generator=g
                )
    
    model = Autoencoder({**params, 'patch_size': best_hyperparams['patch_size']}).to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    patches = extract_patches_2d(S_GT, (best_hyperparams['patch_size'], best_hyperparams['patch_size']), max_patches=params['num_patches'], random_state=42)
    labels = patches.mean(axis=(1, 2))
    
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    # Store losses for visualization
    contrastive_losses = []
    recon_losses = []
    total_losses = []
    
    # Train model
    for epoch in range(best_hyperparams['epochs']):
        total_loss, contrastive_loss, recon_loss = train_epoch(
            model, data_loader, optimizer, labels, device,
            temperature=best_hyperparams['temperature'], lambda_recon=best_hyperparams['lambda_recon'],
            loss_type='generalized', kernel='rbf', sigma_kernel=1
        )
        
        # Store losses for plotting
        contrastive_losses.append(contrastive_loss)
        recon_losses.append(recon_loss)
        total_losses.append(total_loss)
        
        print(f"Epoch {epoch + 1}/{best_hyperparams['epochs']} - Total Loss: {total_loss:.4f}, Contrastive Loss: {contrastive_loss:.4f}, Recon Loss: {recon_loss:.4f}", flush=True)
    
    # Plot losses
    plot_losses(contrastive_losses, recon_losses, total_losses, dataset_name)
    
    # Extract endmembers
    endmembers = F.relu(model.decoder.output_layer.weight_raw).detach().cpu().numpy()
    
    if endmembers.shape[2] > 1:
        endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
    else:
        endmembers = np.squeeze(endmembers)
    
    predicted_endmembers = endmembers.T
    true_endmembers = GT
    
    # Calculate SAD values
    order_dict, _ = order_endmembers(true_endmembers, predicted_endmembers)
    reordered_predicted_endmembers = predicted_endmembers[[order_dict[k] for k in sorted(order_dict.keys())]]
    
    run_sad_values = [numpy_SAD(reordered_predicted_endmembers[j, :], true_endmembers[j, :]) for j in range(true_endmembers.shape[0])]
    
    print(f"Dataset: {dataset_name}")
    print(f"Mean SAD: {np.mean(run_sad_values)}")
    print(run_sad_values)
    
    # Plot results
    plot_endmembers(true_endmembers, reordered_predicted_endmembers)
    
    del model, optimizer, data_loader, hsi_dataset, data, GT, S_GT, patches, labels
    torch.cuda.empty_cache()