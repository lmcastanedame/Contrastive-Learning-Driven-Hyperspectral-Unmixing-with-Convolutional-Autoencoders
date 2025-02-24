import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import torch
from augmentations import random_crop_resize, random_brightness_contrast, random_gaussian_blur
from losses import NTXenLoss, GeneralizedSupervisedNTXenLossWithReconstruction

def apply_augmentation(image, alpha=None, crop=0.75, sigma=1.0):
    """
    Apply the specified augmentation to the image.
    
    Args:
        image (torch.Tensor): The input image [C, H, W].
        alpha (float): Brightness/contrast scaling factor for jitter.
        crop (float): Crop ratio for cropping transformations.
        sigma (float): Maximum sigma for Gaussian blur.

    Returns:
        torch.Tensor: Augmented image.
        float: Alpha value (used for jitter).
    """
    
    if random.random() < 0.5:
        image = random_gaussian_blur(image, prob=1, max_sigma=sigma)
        return image, 1.0  # Alpha is 1 (no scaling)
    
    elif random.random() < 0.5:
        image = random_crop_resize(image, prob=1, crop_ratio=crop)
        image = random_gaussian_blur(image, prob=1, max_sigma=sigma)
        return image, 1.0  # Alpha is 1 (no scaling)
    
    elif random.random() < 0.5:
        if alpha is None:
            raise ValueError("Alpha must be provided for jitter augmentations.")
        image = random_crop_resize(image, prob=1, crop_ratio=crop)
        image = random_brightness_contrast(image, prob=1, alpha=alpha)
        image = random_gaussian_blur(image, prob=1, max_sigma=sigma)
        return image, alpha  # Return the alpha used
    
    return image, 1.0  # Default: no augmentation

# Load HSI dataset
def load_HSI(path):
    """
    Load hyperspectral image (HSI) data from a given file path.
    Parameters:
        path (str): The file path of the HSI data.
    Returns:
        tuple: A tuple containing the following arrays:
            - Y (ndarray): The HSI data array.
            - GT (ndarray): The ground truth array.
            - S_GT (ndarray or None): The spatial ground truth array, if available.
    """    
    
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
    
    return Y, GT, S_GT

class HSI_Dataset(Dataset):
    def __init__(self, hsi, patch_size, patch_number, alpha_range=None, random_state=42):
        """
        hsi: numpy array of shape [H, W, C]
        patch_size: int, size of each square patch
        patch_number: int, number of patches to extract
        alpha_range: tuple, range of alpha values (min, max) for scaling in jitter.
        augmentation_type: str, type of augmentation ('blur', 'crop_blur', 'crop_jitter_blur').
        random_state: int, for reproducibility.
        """
        self.patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number, random_state=random_state)
        self.alpha_range = alpha_range
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]  # [patch_size, patch_size, C]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()  # [C, H, W]
        
        # Determine alpha for jitter if needed
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])

        # Apply the specified augmentation
        aug1, alpha1 = apply_augmentation(patch.clone(), alpha=alpha)
        aug2, alpha2 = apply_augmentation(patch.clone(), alpha=alpha)
        
        # Stack augmented patches
        stacked = torch.stack([aug1, aug2], dim=0)  # [2, C, H, W]
        return stacked, (alpha1, alpha2)  # Return alphas for later use

# Train function
def train_epoch(
    model, data_loader, optimizer, labels, device, temperature, 
    lambda_recon, loss_type, kernel, sigma_kernel
):
    """
    Trains the model for one epoch using the given data loader and optimizer.
    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        optimizer (Optimizer): The optimizer used for training.
        labels (Tensor): The labels associated with the training data.
        device (str): The device to be used for training (e.g., 'cpu', 'cuda').
        temperature (float): The temperature parameter for contrastive loss.
        lambda_recon (float): The weight parameter for the reconstruction loss.
        loss_type (str): The type of loss to be used ('generalized', 'contrastive', 'reconstruction').
        kernel (str): The kernel function to be used for contrastive loss.
        sigma_kernel (float): The sigma parameter for the kernel function.
    Returns:
        tuple: A tuple containing the total loss, contrastive loss, and reconstruction loss for the epoch.
    """
    
    model.train()
    nb_batch = len(data_loader)
    total_loss = 0
    contrastive_loss_total = 0
    reconstruction_loss_total = 0
    
    for inputs, alphas in data_loader:
        inputs = inputs.to(device)
        alphas = alphas.to(device)
        
        # Extract relevant labels for the batch if using generalized loss
        batch_labels = labels[:inputs.size(0)] if loss_type == 'generalized' else None

        optimizer.zero_grad()
        
        reconstructed_i, z_i = model(inputs[:, 0, :], alphas[:,0].shape)  # First view
        reconstructed_j, z_j = model(inputs[:, 1, :], alphas[:,1].shape)  # Second view
        
        # Use the selected loss
        loss, contrastive_loss, reconstruction_loss = select_loss(
            loss_type=loss_type,
            z_i=z_i, z_j=z_j,
            reconstructed_i=reconstructed_i, reconstructed_j=reconstructed_j,
            inputs_i=inputs[:, 0, :], inputs_j=inputs[:, 1, :],
            labels=batch_labels,
            temperature=temperature,
            lambda_recon=lambda_recon,
            kernel=kernel,
            sigma=sigma_kernel
        )
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += float(loss) / nb_batch
        contrastive_loss_total += float(contrastive_loss) / nb_batch
        reconstruction_loss_total += float(reconstruction_loss) / nb_batch
        
    return total_loss, contrastive_loss_total, reconstruction_loss_total

# Plotting functions
def plot_endmembers(endmembers):
    """
    Plots the endmembers.
    Parameters:
        endmembers (ndarray): The endmembers to be plotted.
    Returns:
        None
    """
    
    endmembers = endmembers / endmembers.max()
    num_endmembers = endmembers.shape[0]
    for i in range(num_endmembers):
        plt.plot(endmembers[i, :])
    plt.show()

def select_loss(
    loss_type='ntxen',  # 'ntxen' or 'generalized'
    z_i=None, z_j=None, reconstructed_i=None, reconstructed_j=None,
    inputs_i=None, inputs_j=None, labels=None,
    temperature=0.5, lambda_recon=0.9, kernel='dot', sigma=1.0
):
    
    """
    Selects and returns the appropriate loss function based on the given loss type.
    Parameters:
        loss_type (str): The type of loss function to select. Can be either 'ntxen' or 'generalized'.
        z_i (Tensor): The tensor representation of the first input.
        z_j (Tensor): The tensor representation of the second input.
        reconstructed_i (Tensor): The reconstructed version of the first input.
        reconstructed_j (Tensor): The reconstructed version of the second input.
        inputs_i (Tensor): The original first input.
        inputs_j (Tensor): The original second input.
        labels (Tensor): The labels for the generalized loss function. Required only for 'generalized' loss type.
        temperature (float): The temperature parameter for the loss function. Default is 0.5.
        lambda_recon (float): The weight parameter for the reconstruction loss. Default is 0.9.
        kernel (str): The kernel type for the generalized loss function. Default is 'dot'.
        sigma (float): The sigma parameter for the generalized loss function. Default is 1.0.
    Returns:
        Loss: The selected loss function based on the given loss type.
    Raises:
        ValueError: If the given loss type is unknown.
        ValueError: If labels are not provided for the generalized loss function.
    """
    
    if loss_type == 'ntxen':
        return NTXenLoss(
            z_i, z_j,
            reconstructed_i, reconstructed_j,
            inputs_i, inputs_j,
            temperature=temperature,
            lambda_recon=lambda_recon
        )
    elif loss_type == 'generalized':
        if labels is None:
            raise ValueError("Labels must be provided for the generalized loss.")
        return GeneralizedSupervisedNTXenLossWithReconstruction(
            z_i, z_j,
            reconstructed_i, reconstructed_j,
            inputs_i, inputs_j,
            labels,
            temperature=temperature,
            lambda_recon=lambda_recon,
            kernel=kernel,
            sigma=sigma
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")