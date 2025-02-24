from scipy.ndimage import gaussian_filter
from skimage import transform as sk_tf
from collections import namedtuple
import numpy as np
import numbers
import random
import torch

def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symmetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boundaries.")
    return tuple(obj)
    
class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.clone()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- '+trf.__str__()
        return s


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, arr):
        arr_mean = arr.mean()
        arr_std = arr.std()
        return self.std * (arr - arr_mean) / (arr_std + self.eps) + self.mean

# New Augmentation Classes
class RandomCropResize(object):
    def __init__(self, crop_ratio=0.95):
        self.crop_ratio = crop_ratio

    def __call__(self, image):
        h, w = image.shape[:2]
        crop_h, crop_w = int(self.crop_ratio * h), int(self.crop_ratio * w)
        top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)
        cropped_image = image[top:top + crop_h, left:left + crop_w, :]
        if isinstance(cropped_image, torch.Tensor):
            cropped_image = cropped_image.cpu().numpy()

        return sk_tf.resize(cropped_image, (h, w), preserve_range=True)


class RandomFlip(object):
    def __init__(self, flip_option=0):
        self.flip_option = flip_option

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        if self.flip_option == 0:
            flip_type = random.choice([1, 2, 3])
        else:
            flip_type = self.flip_option

        if flip_type == 1:  # Horizontal flip
            return np.flip(image, axis=1)
        elif flip_type == 2:  # Vertical flip
            return np.flip(image, axis=0)
        elif flip_type == 3:  # Horizontal + vertical flip
            return np.flip(np.flip(image, axis=1), axis=0)
        return image


class RandomBrightnessContrast(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, image):
        return self.alpha * image


class RandomGaussianBlur(object):
    def __init__(self, min_kernel_size=3, max_kernel_size=7, min_sigma=0.1, max_sigma=1.0):
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, image):
        kernel_size = random.choice(range(self.min_kernel_size, self.max_kernel_size + 1, 2))
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        return gaussian_filter(image, sigma=sigma)


class CropBlur(object):
    def __init__(self, crop_ratio=0.95, min_sigma=0.1, max_sigma=1.0):
        self.crop = RandomCropResize(crop_ratio=crop_ratio)
        self.blur = RandomGaussianBlur(min_sigma=min_sigma, max_sigma=max_sigma)

    def __call__(self, image):
        image = self.crop(image)
        image = self.blur(image)
        return image
    
    
class CropOnly(object):
    def __init__(self, crop_ratio=0.95):
        self.crop = RandomCropResize(crop_ratio=crop_ratio)

    def __call__(self, image):
        return self.crop(image)
    
    
class CropFlipBlur(object):
    def __init__(self, crop_ratio=0.95, flip_option=0, min_sigma=0.1, max_sigma=1.0):
        self.crop = RandomCropResize(crop_ratio=crop_ratio)
        self.flip = RandomFlip(flip_option=flip_option)
        self.blur = RandomGaussianBlur(min_sigma=min_sigma, max_sigma=max_sigma)

    def __call__(self, image):
        image = self.crop(image)
        image = self.flip(image)
        image = self.blur(image)
        return image
    
    
class FlipOnly(object):
    def __init__(self, flip_option=0):
        self.flip = RandomFlip(flip_option=flip_option)

    def __call__(self, image):
        return self.flip(image)
    
    
    
    
    