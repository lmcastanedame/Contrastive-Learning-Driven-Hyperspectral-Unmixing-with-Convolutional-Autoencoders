import random
import torch
import torch.nn.functional as F


# Helper functions
def random_crop_resize(image, prob=0, crop_ratio=0.95):
    """
    Randomly crops and resizes an image.
    Args:
        image (numpy.ndarray): The input image.
        prob (float): The probability of applying the crop and resize operation. Default is 0.
        crop_ratio (float): The ratio of the cropped image size to the original image size. Default is 0.95.
    Returns:
        numpy.ndarray: The cropped and resized image.
    """
    
    if prob == 1:
        h, w, c = image.shape
        crop_h, crop_w = int(crop_ratio * h), int(crop_ratio * w)
        
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop
        cropped_image = image[top:top + crop_h, left:left + crop_w, :]
        
        # Resize back to original size
        # interpolate expects [N, C, H, W], so rearrange dimensions
        cropped_image = cropped_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, crop_h, crop_w]
        resized = F.interpolate(cropped_image, size=(h, w), mode='bilinear', align_corners=False)
        image = resized.squeeze(0).permute(1, 2, 0)  # back to [H, W, C]
    return image

def random_flip(image, prob=0, flip_option=0):
    """
    image: torch.Tensor of shape [H, W, C]
    flip_option: 0 - random choice among (horizontal, vertical, both)
                 1 - horizontal
                 2 - vertical
                 3 - both horizontal and vertical
    """
    if prob == 1:
        if flip_option == 0:
            flip_type = random.choice([1, 2, 3])
        else:
            flip_type = flip_option

        # For flipping, we can flip by indexing directly:
        # Horizontal flip: image = image[:, ::-1, :]
        # Vertical flip: image = image[::-1, :, :]

        if flip_type == 1:
            # Horizontal flip
            image = image[:, torch.arange(image.shape[1]-1, -1, -1), :]
        elif flip_type == 2:
            # Vertical flip
            image = image[torch.arange(image.shape[0]-1, -1, -1), :, :]
        elif flip_type == 3:
            # Both flips
            image = image[torch.arange(image.shape[0]-1, -1, -1), :, :]  # vertical
            image = image[:, torch.arange(image.shape[1]-1, -1, -1), :]  # horizontal
    return image

def random_brightness_contrast(image, prob=0, alpha=None):
    """
    image: torch.Tensor of shape [H, W, C]
    alpha: scaling factor for brightness/contrast
    """
    if prob == 1 and alpha is not None:
        image = image * alpha
    return image

# Gaussian kernel cache
kernel_cache = {}

def gaussian_kernel(size: int, sigma: float):
    """
    Generates a 2D Gaussian kernel using PyTorch.
    Args:
        size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
    Returns:
        torch.Tensor: The generated 2D Gaussian kernel.
    """
    
    # Create a 2D Gaussian kernel using PyTorch
    coords = torch.arange(size) - size//2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    x = x.float()
    y = y.float()
    kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def get_gaussian_kernel(size, sigma):
    """
    Retrieves or generates a Gaussian kernel of a given size and sigma.
    Parameters:
        size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
    Returns:
        numpy.ndarray: The generated Gaussian kernel.
    """
    
    key = (size, sigma)
    if key not in kernel_cache:
        kernel_cache[key] = gaussian_kernel(size, sigma)
    return kernel_cache[key]

def random_gaussian_blur(image, prob=0, min_kernel_size=3, max_kernel_size=7, min_sigma=0.1, max_sigma=1.0):
    """
    image: torch.Tensor [H, W, C]
    Applies Gaussian blur to the image by convolving each channel separately.
    """
    if prob == 1:
        kernel_size = random.choice(range(min_kernel_size, max_kernel_size+1, 2))
        sigma = random.uniform(min_sigma, max_sigma)
        kernel = get_gaussian_kernel(kernel_size, sigma)  # [kernel_size, kernel_size]
        
        # Prepare for conv2d
        # image: [H, W, C] -> [N=1, C, H, W]
        image = image.permute(2, 0, 1).unsqueeze(0)
        
        # kernel: [K, K]
        # For conv2d we need [out_channels, in_channels, K, K]
        # We'll apply the same kernel to each channel independently using groups
        c = image.shape[1]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        kernel = kernel.to(image.device)
        kernel = kernel.expand(c, 1, kernel_size, kernel_size)  # [C, 1, K, K]

        # Convolution with groups = C ensures each channel is filtered separately
        image = F.conv2d(image, weight=kernel, bias=None, stride=1, padding=kernel_size//2, groups=c)
        
        # Back to [H, W, C]
        image = image.squeeze(0).permute(1, 2, 0)
    return image