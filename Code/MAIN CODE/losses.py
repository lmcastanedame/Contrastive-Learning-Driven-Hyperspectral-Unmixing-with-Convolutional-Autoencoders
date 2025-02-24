import torch
import torch.nn.functional as func
import torch.nn as nn

# SAD Loss
def SAD(y_true, y_pred):
    """
    Calculates the Sum of Absolute Differences (SAD) between two tensors.
    Args:
        y_true (torch.Tensor): The true tensor.
        y_pred (torch.Tensor): The predicted tensor.
    Returns:
        torch.Tensor: The mean SAD value.
    """
    
    y_true = nn.functional.normalize(y_true, dim=1)
    y_pred = nn.functional.normalize(y_pred, dim=1)
    A = (y_true * y_pred).sum(dim=1)
    sad = torch.acos(A)
    return sad.mean()

def GeneralizedSupervisedNTXenLossWithReconstruction(
    z_i, z_j,
    reconstructed_i, reconstructed_j,
    inputs_i, inputs_j,
    labels, temperature, lambda_recon, kernel, sigma, return_logits=False
):
    """
    Calculates the Generalized Supervised NT-Xent Loss with Reconstruction.
    Args:
        z_i (torch.Tensor): Embeddings of view i. Shape [N, D].
        z_j (torch.Tensor): Embeddings of view j. Shape [N, D].
        reconstructed_i (torch.Tensor): Reconstructed inputs of view i. Shape [N, C, H, W].
        reconstructed_j (torch.Tensor): Reconstructed inputs of view j. Shape [N, C, H, W].
        inputs_i (torch.Tensor): Inputs of view i. Shape [N, C, H, W].
        inputs_j (torch.Tensor): Inputs of view j. Shape [N, C, H, W].
        labels (torch.Tensor): Labels of the samples. Shape [N, num_endmembers].
        temperature (float): Temperature parameter for the contrastive loss.
        lambda_recon (float): Weighting factor for the reconstruction loss.
        kernel (str): Kernel type for the multi-kernel computation. Can be 'dot' or 'rbf'.
        sigma (float): Standard deviation for the RBF kernel.
        return_logits (bool, optional): Whether to return the logits. Defaults to False.
    Returns:
        tuple: A tuple containing the total loss, contrastive loss, reconstruction loss, 
               similarity matrix, and correct pairs (if return_logits is True).
    Raises:
        ValueError: If the kernel type is not supported.
    Notes:
        - This function assumes that the labels are one-hot encoded.
        - The contrastive loss is calculated using the NT-Xent loss formula.
        - The reconstruction loss is calculated using the Sum of Absolute Differences (SAD) metric.
        - The total loss is a weighted combination of the contrastive loss and the reconstruction loss.
        - The similarity matrix is computed by concatenating the pairwise cosine similarities of the embeddings.
        - The kernel weights are computed based on the labels and the chosen kernel type.
        - The kernel weights are then used to combine the similarity matrix and compute the contrastive loss.
    """
    
    INF = 1e8
    N = z_i.shape[0]  # Number of samples

    # Normalize embeddings
    z_i = func.normalize(z_i, p=2, dim=1)
    z_j = func.normalize(z_j, p=2, dim=1)

    # Compute pairwise cosine similarities
    sim_zii = (z_i @ z_i.T) / temperature  # [N, N]
    sim_zjj = (z_j @ z_j.T) / temperature  # [N, N]
    sim_zij = (z_i @ z_j.T) / temperature  # [N, N]

    # Mask self-similarities
    sim_zii = sim_zii - INF * torch.eye(N, device=z_i.device)
    sim_zjj = sim_zjj - INF * torch.eye(N, device=z_j.device)

    # Multi-kernel for endmembers
    num_endmembers = labels.shape[1]  # Assuming labels are [N, num_endmembers]
    kernel_weights = torch.ones((N, N), device=z_i.device)  # Start with uniform weights

    for e in range(num_endmembers):
        y_e = labels[:, e].view(N, 1)  # Extract endmember values
        if kernel == 'dot':
            kernel_fn = lambda y1, y2: y1 @ y2.T
        elif kernel == 'rbf':
            kernel_fn = lambda y1, y2: torch.exp(-torch.cdist(y1, y2) ** 2 / (2 * sigma ** 2))
        kernel_weights *= kernel_fn(y_e, y_e)  # Combine kernels

    # Expand for both views
    kernel_weights = kernel_weights.repeat(2, 2)  # Ensure shape is [2N, 2N]
    kernel_weights = kernel_weights * (1 - torch.eye(2 * N, device=z_i.device))  # Remove diagonal
    kernel_weights = kernel_weights / kernel_weights.sum(dim=1, keepdim=True)  # Normalize

    # Combined similarity matrix
    sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0)
    log_sim_Z = func.log_softmax(sim_Z, dim=1)

    # Contrastive loss
    contrastive_loss = -1.0 / N * (kernel_weights * log_sim_Z).sum()

    # Reconstruction loss using SAD
    recon_loss_i = SAD(reconstructed_i, inputs_i)
    recon_loss_j = SAD(reconstructed_j, inputs_j)
    reconstruction_loss = (recon_loss_i + recon_loss_j) / 2  # Average from all views

    # Total loss
    total_loss = (1 - lambda_recon) * contrastive_loss + lambda_recon * reconstruction_loss

    if return_logits:
        correct_pairs = torch.arange(N, device=z_i.device).long()
        return total_loss, contrastive_loss, reconstruction_loss, sim_zij, correct_pairs

    return total_loss, contrastive_loss, reconstruction_loss

def NTXenLoss(z_i, z_j, reconstructed_i, reconstructed_j, inputs_i, inputs_j, temperature, lambda_recon):
    """
    Combined loss for contrastive learning and reconstruction in a hyperspectral unmixing task.

    Args:
        z_i, z_j: Embeddings from the encoder for both augmented views.
        reconstructed_i, reconstructed_j: Reconstructed signals for both augmented views.
        inputs_i, inputs_j: Original signals corresponding to both views.
        temperature: Temperature scaling for contrastive loss.
        lambda_recon: Weight for the reconstruction loss.

    Returns:
        total_loss: Combined loss.
        contrastive_loss: Loss from contrastive learning.
        reconstruction_loss: Loss from reconstruction error.
    """
    N = len(z_i)
    INF = 1e8

    # Normalize embeddings
    z_i = func.normalize(z_i, p=2, dim=-1)  # [N, D]
    z_j = func.normalize(z_j, p=2, dim=-1)  # [N, D]

    # Compute pairwise cosine similarities
    sim_zii = (z_i @ z_i.T) / temperature  # [N, N]
    sim_zjj = (z_j @ z_j.T) / temperature  # [N, N]
    sim_zij = (z_i @ z_j.T) / temperature  # [N, N]

    # Mask self-similarities
    sim_zii = sim_zii - INF * torch.eye(N, device=z_i.device)
    sim_zjj = sim_zjj - INF * torch.eye(N, device=z_j.device)

    # Create positive pairs (diagonal elements of sim_zij)
    correct_pairs = torch.arange(N, device=z_i.device).long()

    # Contrastive loss
    loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
    loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)
    contrastive_loss = (loss_i + loss_j) / 2  # Average from both views

    # Reconstruction loss
    recon_loss_i = SAD(reconstructed_i, inputs_i)
    recon_loss_j = SAD(reconstructed_j, inputs_j)
    reconstruction_loss = (recon_loss_i + recon_loss_j) / 2  # Average from both views

    # Combined loss
    total_loss = (1 - lambda_recon) * contrastive_loss + lambda_recon * reconstruction_loss

    return total_loss, contrastive_loss, reconstruction_loss