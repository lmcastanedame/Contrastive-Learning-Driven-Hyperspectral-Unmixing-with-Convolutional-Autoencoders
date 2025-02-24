from torch.utils.data import Dataset
import numpy as np
import torch
from augmentations import Transformer, CropBlur, CropOnly, CropFlipBlur, FlipOnly, Normalize
from sklearn.feature_extraction.image import extract_patches_2d
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset

def load_HSI(path):
        data = sio.loadmat(path)
        Y = np.asarray(data['Y'], dtype=np.float32)
        GT = np.asarray(data['GT'], dtype=np.float32)
        S_GT = np.asarray(data['S_GT'], dtype=np.float32)
        if Y.shape[0] < Y.shape[1]:
            Y = Y.transpose()
        Y = Y / np.max(Y.flatten())
        n_bands = Y.shape[1]
        n_rows = data['lines'].item()
        n_cols = data['cols'].item()
        Y = np.reshape(Y, (n_cols, n_rows, n_bands))
        return Y, GT, S_GT
class HyperspectralDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        """
        A dataset for patch-based hyperspectral image learning.
        Parameters:
        - config: Config object with dataset configuration.
        - training: bool, whether to load training data.
        - validation: bool, whether to load validation data.
        - patch_size: tuple, the spatial dimensions of the patches.
        - stride: int, the stride for patch extraction.
        """
        super().__init__(*args, **kwargs)
        assert training != validation, "Dataset must be either training or validation, not both."
        self.config = config
        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)

        # Configure transformations
        if config.tf == "all_tf":
            self.transforms.register(CropBlur(crop_ratio=0.95, min_sigma=0.1, max_sigma=1.0), probability=0.5)
            self.transforms.register(CropOnly(crop_ratio=0.95), probability=0.5)
            self.transforms.register(CropFlipBlur(crop_ratio=0.95, flip_option=0, min_sigma=0.1, max_sigma=1.0), probability=0.5)
            self.transforms.register(FlipOnly(flip_option=0), probability=0.5)
            
        elif config.tf == "crop_blur":
            self.transforms.register(CropBlur(crop_ratio=0.95, min_sigma=0.1, max_sigma=1.0), probability=1)
            
        elif config.tf == "crop_only":
            self.transforms.register(CropOnly(crop_ratio=0.95), probability=1)
            
        elif config.tf == "crop_flip_blur":
            self.transforms.register(CropFlipBlur(crop_ratio=0.95, flip_option=0, min_sigma=0.1, max_sigma=1.0), probability=1)
            
        elif config.tf == "flip_only":
            self.transforms.register(FlipOnly(flip_option=0), probability=1)

        # Load and preprocess the data
        dataset_path = config.data_train
        self.image, self.GT, self.S_GT = load_HSI(dataset_path)
        random_state = 42
        
        if training:
            self.data = HSI_Dataset(self.image, config.patch_size, config.num_patches, random_state=random_state)
            self.patches = HSI_Dataset(self.S_GT, config.patch_size, config.num_patches, random_state=random_state)
            self.labels = torch.zeros(config.num_patches, config.num_endmembers)
            for i in range(config.num_patches):
                patch = self.patches[i]  # patch shape: [E, H, W]
                max_abundance = patch.reshape(config.num_endmembers, -1).max(dim=1)[0]
                
                self.labels[i, :] = max_abundance
                
        elif validation:
            self.data = HSI_Dataset(self.image, config.patch_size, config.num_patches, random_state=random_state)
            self.patches = HSI_Dataset(self.S_GT, config.patch_size, config.num_patches, random_state=random_state)
            self.labels = torch.zeros(config.num_patches, config.num_endmembers)
            for i in range(config.num_patches):
                patch = self.patches[i]  # patch shape: [E, H, W]
                max_abundance = patch.reshape(config.num_endmembers, -1).max(dim=1)[0]
                
                self.labels[i, :] = max_abundance

        sample_patch = self.data[0]  # Get the first patch, shape: [C, H, W]
        assert sample_patch.shape == tuple(config.input_size), \
            f"Each patch must have shape {config.input_size}, but got {sample_patch.shape}"

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        np.random.seed()
        x1 = self.transforms(self.data[idx])
        x2 = self.transforms(self.data[idx])
        labels = self.labels[idx]
        x = np.stack((x1, x2), axis=0)

        return (x, labels)

    def __len__(self):
        return len(self.data)
    
class HSI_Dataset(Dataset):
    def __init__(self, hsi, patch_size, patch_number, random_state):
        self.patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number, random_state=random_state)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return torch.from_numpy(patch).permute(2, 0, 1)  # Convert to CHW format
    
    
    