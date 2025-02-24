import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

# Disable eager execution equivalent is not required in PyTorch

# Load HSI dataset
def load_HSI(path):
    data = sio.loadmat(path)
    Y = np.asarray(data['Y'], dtype=np.float32)
    GT = np.asarray(data['GT'], dtype=np.float32)
    if Y.shape[0] < Y.shape[1]:
        Y = Y.transpose()
    Y = Y / np.max(Y.flatten())
    n_bands = Y.shape[1]
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    Y = np.reshape(Y, (n_cols, n_rows, n_bands))
    return Y, GT

# Training patches dataset
class HSI_Dataset(Dataset):
    def __init__(self, hsi, patch_size, patch_number):
        self.patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return torch.from_numpy(patch).permute(2, 0, 1)  # Convert to CHW format

# Alpha Scaling Layer
class AlphaScalingLayer(nn.Module):
    """
    Scales the abundances by alpha to ensure coherence.
    """
    def __init__(self):
        super(AlphaScalingLayer, self).__init__()

    def forward(self, abundances, alpha):
        # Ensure alpha is of floating type
        alpha = alpha.float()
        
        # Reshape alpha to broadcast correctly with abundances
        alpha = alpha.view(-1, 1, 1, 1)
        
        # Multiply abundances by alpha
        scaled_abundances = abundances * alpha
        return scaled_abundances

# Sum to One Layer
class SumToOne(nn.Module):
    def __init__(self, scale):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.nn.functional.softmax(self.scale * x, dim=1)
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.hidden_layer_one = nn.Conv2d(in_channels=params.n_bands,
                                          out_channels=params.e_filters,
                                          kernel_size=params.e_size,
                                          stride=1, padding='same')
        self.hidden_layer_two = nn.Conv2d(in_channels=params.e_filters,
                                          out_channels=params.num_endmembers,
                                          kernel_size=1, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(params.e_filters)  # Initialize BatchNorm in constructor
        self.bn2 = nn.BatchNorm2d(params.num_endmembers)
        self.asc_layer = SumToOne(params.scale)
        
        nn.init.normal_(self.hidden_layer_one.weight, mean=0.0, std=0.03)
        nn.init.normal_(self.hidden_layer_two.weight, mean=0.0, std=0.03)

    def forward(self, x):
        x = self.hidden_layer_one(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.bn1(x)  # Use initialized BatchNorm
        x = nn.Dropout2d(0.2)(x)
        x = self.hidden_layer_two(x)
        x = nn.LeakyReLU(0.02)(x)
        x = self.bn2(x)  # Use initialized BatchNorm
        x = nn.Dropout2d(0.2)(x)
        x = self.asc_layer(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.output_layer = nn.Conv2d(in_channels=params.num_endmembers,
                                      out_channels=params.d_filters,
                                      kernel_size=params.d_size,
                                      stride=1, padding='same')
        
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.03)

    def forward(self, x):
        x = nn.ReLU()(x)
        recon = self.output_layer(x)
        return recon

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.alpha_scaling_layer = AlphaScalingLayer()

    def forward(self, x): #, alpha):
        abundances = self.encoder(x)
        # scaled_abundances = self.alpha_scaling_layer(abundances, alpha)
        reconstructed = self.decoder(abundances)
        
        # Convert to a single vector per sample: [N, num_endmembers]
        embedding = F.adaptive_avg_pool2d(abundances, (1, 1))   # [N, num_endmembers, 1, 1]
        embedding = torch.flatten(embedding, start_dim=1)       # [N, num_endmembers]
        
        # endmembers = self.decoder.output_layer.weight.detach().cpu().numpy()
        # if endmembers.shape[2] > 1:
        #     endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        # else:
        #     endmembers = np.squeeze(endmembers)
        
        return reconstructed, embedding

# SAD Loss
def SAD(y_true, y_pred):
    y_true = nn.functional.normalize(y_true, dim=1)
    y_pred = nn.functional.normalize(y_pred, dim=1)
    A = (y_true * y_pred).sum(dim=1)
    sad = torch.acos(A)
    return sad.mean()

# Train function
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)  # Move batch to the same device as the model
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Plotting functions
def plot_endmembers(endmembers):
    endmembers = endmembers / endmembers.max()
    num_endmembers = endmembers.shape[0]
    for i in range(num_endmembers):
        plt.plot(endmembers[i, :])
    plt.show()

# Main training loop
if __name__ == '__main__':
    # Hyperparameters
    params = {
        'n_bands': 156, 'e_filters': 48, 'e_size': 3, 'd_filters': 156,
        'd_size': 13, 'num_endmembers': 3, 'scale': 3,
        'patch_size': 40, 'num_patches': 250, 'batch_size': 15, 'epochs': 320,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = "../Datasets/Samson.mat"
    data, GT = load_HSI(dataset_path)

    # Prepare dataset and dataloader
    hsi_dataset = HSI_Dataset(data, params['patch_size'], params['num_patches'])
    data_loader = DataLoader(hsi_dataset, batch_size=params['batch_size'], shuffle=True)

    # Initialize model, loss, and optimizer
    model = Autoencoder(params).to(device)
    criterion = SAD
    optimizer = optim.RMSprop(model.parameters(), lr=0.003)

    # Train the model
    for epoch in range(params['epochs']):
        epoch_loss = train_epoch(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {epoch_loss:.4f}")

    # Save and plot results
    torch.save(model.state_dict(), 'autoencoder_model.pth')