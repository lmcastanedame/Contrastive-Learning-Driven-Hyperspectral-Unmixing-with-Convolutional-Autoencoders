import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, SpatialDropout2D, Layer
from tensorflow.keras import optimizers
from scipy import io as sio
import numpy as np
import os
import random
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
tf.config.optimizer.set_jit(False)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Ensure memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Function to set seeds for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
# Function to load hyperspectral image (HSI) and reference endmembers
def load_HSI(path):
    """
    Loads the hyperspectral image (HSI) and reference endmembers from a .mat file.
    """
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = hdf.File(path, 'r')  # Handle HDF5 if .mat is unsupported

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

# Function to extract patches for training
def training_input_fn_tf(hsi, patch_size, patch_number, alpha_value=None):
    """
    Extracts patches for training from the hyperspectral image.
    Optionally assigns a scaling factor (alpha) to each patch.
    """
    hsi = tf.convert_to_tensor(hsi, dtype=tf.float32)
    patches = tf.image.extract_patches(
        images=tf.expand_dims(hsi, axis=0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patches = tf.reshape(patches, [-1, patch_size, patch_size, hsi.shape[-1]])[:patch_number]
    patches = patches.numpy()

    if alpha_value is not None:
        alphas = np.full((len(patches),), alpha_value, dtype=np.float32)
        return patches, alphas
    return patches

# ==================== Augmentation functions ====================
def random_crop_resize(image, prob=0, crop_ratio=0.95):
    if prob == 1:
        h, w = image.shape[:2]
        crop_h, crop_w = int(crop_ratio * h), int(crop_ratio * w)
        top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)
        cropped_image = image[top:top + crop_h, left:left + crop_w, :]
        image = tf.image.resize(cropped_image, [h, w])
    return image

def random_flip(image, prob=0, flip_option=0):
    if prob == 1:
        if flip_option == 0:
            flip_type = random.choice([1, 2, 3])
        else:
            flip_type = flip_option
            
        if flip_type == 1:
            image = tf.image.flip_left_right(image)
        elif flip_type == 2:
            image = tf.image.flip_up_down(image)
        elif flip_type == 3:
            image = tf.image.flip_left_right(image)
            image = tf.image.flip_up_down(image)
    return image

def random_brightness_contrast(image, prob=0, alpha=None):
    if prob == 1 and alpha is not None:
        image = alpha * image
    return image

# Gaussian kernel for random blur
kernel_cache = {}

def gaussian_kernel(size: int, sigma: float):
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    y = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(x, y)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[:, :, tf.newaxis, tf.newaxis]

def get_gaussian_kernel(size, sigma):
    key = (size, sigma)
    if key not in kernel_cache:
        kernel_cache[key] = gaussian_kernel(size, sigma)
    return kernel_cache[key]

def random_gaussian_blur(image, prob=0, min_kernel_size=3, max_kernel_size=7, min_sigma=0.1, max_sigma=1.0):
    if prob == 1:
        kernel_size = random.choice(range(min_kernel_size, max_kernel_size + 1, 2))
        sigma = random.uniform(min_sigma, max_sigma)
        kernel = get_gaussian_kernel(kernel_size, sigma)
        channels = tf.split(image, num_or_size_splits=image.shape[-1], axis=-1)
        blurred_channels = [tf.nn.conv2d(channel[tf.newaxis, ...], kernel, strides=[1, 1, 1, 1], padding='SAME')
                            for channel in channels]
        image = tf.concat([tf.squeeze(channel, axis=0) for channel in blurred_channels], axis=-1)
    return image

# Function to apply augmentations
def apply_augmentation(image, p1=0, p2=0, p3=0, p4=0, alpha=None, crop=0.95, flip=3, sigma=2.0):
    image = random_crop_resize(image, prob=p1, crop_ratio=crop)
    image = random_flip(image, prob=p2, flip_option=flip)
    image = random_brightness_contrast(image, prob=p3, alpha=alpha)
    image = random_gaussian_blur(image, prob=p4, max_sigma=sigma)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# ========== Alpha Scaling Layer ==========
class AlphaScalingLayer(Layer):
    """
    Scales the abundances by alpha to ensure coherence.
    """
    def call(self, abundances, alpha):
        alpha = tf.cast(alpha, dtype=tf.float32)
        alpha = tf.reshape(alpha, [-1, 1, 1, 1])  # Reshape to match abundances
        scaled_abundances = abundances * alpha
        return scaled_abundances

# ========== Encoder ==========
class Encoder(Model):
    """
    Encodes input patches into abundance maps.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.conv1 = Conv2D(
            filters=params['e_filters'], kernel_size=params['e_size'], activation=params['activation'],
            padding='same', kernel_initializer=params['initializer'], use_bias=False
        )
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = SpatialDropout2D(0.2)
        self.conv2 = Conv2D(
            filters=params['num_endmembers'], kernel_size=1, activation=params['activation'],
            padding='same', kernel_initializer=params['initializer'], use_bias=False
        )
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = SpatialDropout2D(0.2)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        return tf.nn.softmax(x)  # Enforces the ASC (Abundance Sum-to-One Constraint)

# ========== Decoder ==========
class Decoder(tf.keras.layers.Layer):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.output_layer = tf.keras.layers.Conv2D(
            filters=params['d_filters'], 
            kernel_size=params['d_size'],
            activation='linear',
            kernel_constraint=tf.keras.constraints.non_neg(),
            name='endmembers', strides=1, padding='same',
            kernel_regularizer=None,
            kernel_initializer=params['initializer'], use_bias=False)

    def call(self, code):
        recon = self.output_layer(code)
        return recon

    def getEndmembers(self):
        """
        Retrieves the weights of the decoder's convolutional layer (endmembers).
        """
        return self.output_layer.get_weights()

# ========== Autoencoder ==========
class Autoencoder(Model):
    """
    Combines the Encoder, Decoder, and Alpha Scaling layers.
    """
    def __init__(self, params):
        super().__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.alpha_scaling_layer = AlphaScalingLayer()

    def call(self, patches, alpha, training=False):
        abundances = self.encoder(patches, training=training)
        scaled_abundances = self.alpha_scaling_layer(abundances, alpha)
        reconstructed = self.decoder(scaled_abundances)
        return reconstructed

    def getEndmembers(self):
        """
        Retrieves the 1D spectral signals of the endmembers.
        """
        # Get the weights from the decoder's convolutional layer
        endmembers = self.decoder.getEndmembers()[0]  # Shape: (filter_height, filter_width, num_bands, num_endmembers)

        # Check if spatial dimensions exist and average them
        if len(endmembers.shape) == 4:  # Expecting (filter_height, filter_width, num_bands, num_endmembers)
            endmembers = np.mean(endmembers, axis=(0, 1))  # Average spatial dimensions

        # Ensure shape: (num_endmembers, num_bands)
        endmembers = np.squeeze(endmembers).T  # Transpose to match (num_endmembers, num_bands)

        return endmembers
    
    def getAbundances(self, hsi):
        """
        Predicts abundances for the given HSI.
        """
        hsi = np.expand_dims(hsi, axis=0)
        abundances = self.encoder(hsi, training=False)
        return np.squeeze(abundances)
    
# ========== Loss Function ==========
def SAD(y_true, y_pred):
    """
    Spectral Angle Distance (SAD) loss function.
    """
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return tf.reduce_mean(tf.math.acos(tf.clip_by_value(tf.reduce_sum(y_true * y_pred, axis=-1), -1.0, 1.0)))

# ========== Chunked Training ==========
def train_chunked(model, dataset, alphas, params):
    """
    Trains the model in memory-efficient chunks with early stopping.
    """
    batch_size = params['batch_size']
    chunk_size = params.get('chunk_size', 1000)  # Default chunk size for memory management
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    patience = params.get('patience', 10)  # Early stopping patience
    min_delta = params.get('min_delta', 1e-4)  # Minimum improvement for early stopping

    best_loss = float('inf')
    wait = 0  # Counter for early stopping

    should_stop = False  # Flag for early stopping

    for epoch in range(params['epochs']):
        if should_stop:  # Check if early stopping was triggered
            print(f"Stopping training at epoch {epoch}.", flush=True)
            break

        total_loss = 0
        num_batches = 0

        # Process the dataset in chunks
        for start_idx in range(0, len(dataset), chunk_size):
            chunk_patches = dataset[start_idx:start_idx + chunk_size]
            chunk_alphas = alphas[start_idx:start_idx + chunk_size]
            dataset_chunk = tf.data.Dataset.from_tensor_slices((chunk_patches, chunk_alphas)).batch(batch_size)

            # Train on the current chunk
            for patches, alpha in dataset_chunk:
                with tf.GradientTape() as tape:
                    reconstructed = model(patches, alpha, training=True)
                    loss = SAD(patches, reconstructed)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                total_loss += loss.numpy()
                num_batches += 1

        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{params['epochs']} - Loss: {avg_loss:.4f}", flush=True)

        # Check for early stopping
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            wait = 0  # Reset patience counter
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.", flush=True)
                should_stop = True  # Set stopping flag

# ========== Save Results ==========
def save_results(model, params, filename="results.mat"):
    """
    Saves the endmembers and abundances to a .mat file.
    """
    endmembers = model.getEndmembers()
    abundances = model.getAbundances(params['data'])
    sio.savemat(filename, {'endmembers': endmembers, 'abundances': abundances})
    
# ========== Experiment Runner ==========
def run_experiment(data, alphas, params, result_file="results.mat"):
    """
    Runs the experiment for hyperspectral unmixing.
    Initializes the autoencoder, trains it with chunked processing, and saves results.
    """
    # Initialize the autoencoder
    model = Autoencoder(params)

    # Train the model using chunked processing
    print("Starting training...", flush = True)
    train_chunked(model, data, alphas, params)

    # Save the results
    print("Saving results...", flush = True)
    save_results(model, params, filename=result_file)

    print("Experiment completed successfully!", flush = True)
    
if __name__ == "__main__":

    num_patches = 250   # Number of patches per augmentation
    num_augmentations = 10  # Number of augmentations
    num_runs = 25  # Number of independent runs
    results_folders = {
        './10 images/Results (25 runs, no seed, diff images)/Blur (sigma = [0.1, 2.0])': {
            'p1': 0, 'p2': 0, 'p3': 0, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Blur (sigma = [0.1, 1.0])': {
            'p1': 0, 'p2': 0, 'p3': 0, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop (95)': {
            'p1': 1, 'p2': 0, 'p3': 0, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop (75)': {
            'p1': 1, 'p2': 0, 'p3': 0, 'p4': 0, 'flip' : 0, 'crop' : 0.75, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop (50)': {
            'p1': 1, 'p2': 0, 'p3': 0, 'p4': 0, 'flip' : 0, 'crop' : 0.5, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Vertical Flip': {
            'p1': 0, 'p2': 1, 'p3': 0, 'p4': 0, 'flip' : 1, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Horizontal Flip': {
            'p1': 0, 'p2': 1, 'p3': 0, 'p4': 0, 'flip' : 2, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Mix Flip': {
            'p1': 0, 'p2': 1, 'p3': 0, 'p4': 0, 'flip' : 3, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Random Flip': {
            'p1': 0, 'p2': 1, 'p3': 0, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Jitter': {
            'p1': 0, 'p2': 0, 'p3': 1, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 2.0, 'alpha_range' : (0.8, 1.2)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Blur': {
            'p1': 1, 'p2': 0, 'p3': 0, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Flip': {
            'p1': 1, 'p2': 1, 'p3': 0, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Flip + Blur': {
            'p1': 1, 'p2': 1, 'p3': 0, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (1, 1)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Jitter': {
            'p1': 1, 'p2': 0, 'p3': 1, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (0.8, 1.2)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Jitter + Blur': {
            'p1': 1, 'p2': 0, 'p3': 1, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (0.8, 1.2)
            },
        './10 images/Results (25 runs, no seed, diff images)/Crop + Jitter + Flip': {
            'p1': 1, 'p2': 1, 'p3': 1, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (0.8, 1.2)
            },
        './10 images/Results (25 runs, no seed, diff images)/Jitter + Flip': {
            'p1': 0, 'p2': 1, 'p3': 1, 'p4': 0, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (0.8, 1.2)
            },
        './10 images/Results (25 runs, no seed, diff images)/Jitter + Flip + Blur': {
            'p1': 0, 'p2': 1, 'p3': 1, 'p4': 1, 'flip' : 0, 'crop' : 0.95, 'sigma' : 1.0, 'alpha_range' : (0.8, 1.2)
            },
    }
    
    datasets = {
        # 'Samson': {
        #     'd_filters': 156,
        #     'num_endmembers': 3,
        # },
        # 'Urban4': {
        #     'd_filters': 162,
        #     'num_endmembers': 4,
        # },
        # 'Urban5': {
        #     'd_filters': 162,
        #     'num_endmembers': 5,
        # },
        # 'Urban6': {
        #     'd_filters': 162,
        #     'num_endmembers': 6,
        # }
        # 'Cuprite_fixed': {
        #     'd_filters': 188,
        #     'num_endmembers': 12,
        # },
        'JasperRidge': {
            'd_filters': 198,
            'num_endmembers': 4,
        },
    }

    for dataset in datasets:
        
        # Hyperparameters and model settings
        params = {
            'e_filters': 48,                                            # Number of filters in the encoder
            'e_size': 3,                                                # Encoder kernel size
            'd_filters': datasets[dataset]['d_filters'],                # Number of spectral bands
            'd_size': 13,                                               # Decoder kernel size
            'num_endmembers': datasets[dataset]['num_endmembers'],      # Number of endmembers
            'activation': tf.keras.layers.LeakyReLU(0.02),
            'initializer': tf.keras.initializers.RandomNormal(0.0, 0.3),
            'batch_size': 15,                                           # Training batch size
            'epochs': 320,                                              # Number of training epochs
            'learning_rate': 0.001,                                     # Learning rate
            'chunk_size': 1000,                                         # Chunk size for memory management
            'patience': 10,                                             # Stop if no improvement after 10 epochs
            'min_delta': 1e-4,                                          # Minimum improvement to reset patience
            'data': None                                                # Placeholder for the dataset
        }
        
        print(f"Current Dataset: {dataset}", flush=True)
        
        # Load real hyperspectral data
        dataset_path = "./Datasets/" + dataset + ".mat"  # Path to the dataset
        print("Loading dataset...", flush = True)
        data, GT = load_HSI(dataset_path)  # Load hyperspectral image and ground truth
        
        if dataset == 'Urban4' or dataset == 'Urban5' or dataset == 'Urban6':
            patch_size = 40  # Default for larger datasets
        else:
            patch_size = min(40, data.shape[0] // 2)

        for folder, params_dict in results_folders.items():
            # Create result folder if not exists
            os.makedirs(os.path.join(dataset, folder), exist_ok=True)

            for run in range(num_runs):
                print(f"\nFolder: {folder}, Run: {run + 1}/{num_runs}", flush = True)
                augmented_patches = []
                augmented_alphas = []

                # Apply augmentations and extract patches
                for _ in range(num_augmentations):
                    # Generate a random alpha value
                    alpha_range = params_dict['alpha_range']
                    alpha_value = np.random.uniform(alpha_range[0], alpha_range[1])

                    # Apply augmentations
                    augmented_image = apply_augmentation(
                        data,
                        p1=params_dict['p1'],       # Crop and resize
                        p2=params_dict['p2'],       # Random flip
                        p3=params_dict['p3'],       # Random brightness adjustment
                        p4=params_dict['p4'],       # Gaussian blur
                        alpha=alpha_value,          # Brightness adjustment via alpha
                        crop=params_dict['crop'],   # Crop ratio
                        flip=params_dict['flip'],   # Vertical, horizontal or mix flip
                        sigma=params_dict['sigma'], # Maximum sigma for gaussian kernel
                    )

                    # Extract patches and assign alphas
                    patches, alphas = training_input_fn_tf(augmented_image, patch_size, num_patches, alpha_value)
                    augmented_patches.extend(patches)
                    augmented_alphas.extend(alphas)

                # Convert to numpy arrays
                augmented_patches = np.array(augmented_patches)
                augmented_alphas = np.array(augmented_alphas)

                # Update dataset in parameters
                params['data'] = data

                # Run the experiment
                print("Running the experiment...", flush = True)

                # Save results for this run
                result_file = os.path.join(dataset, folder, f"results_run_{run + 1}.mat")
                
                # Call run_experiment with the result file for each run
                run_experiment(augmented_patches, augmented_alphas, params, result_file=result_file)

            print(f"Results saved in folder: {folder}", flush = True)
            
        print(f"Finished processing {dataset}. Cleaning up memory...", flush=True)
        tf.keras.backend.clear_session()
        gc.collect()