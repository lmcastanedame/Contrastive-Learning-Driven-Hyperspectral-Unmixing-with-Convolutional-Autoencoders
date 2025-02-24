PRETRAINING = 1
FINE_TUNING = 0

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, f"Unknown mode: {mode}"

        self.mode = mode

        if self.mode == PRETRAINING:
            
            # Autoencoder Configuration
            self.e_filters = 48
            self.e_size = 3
            self.d_filters = 156
            self.d_size = 13
            self.scale = 3
            self.patch_size = 40
            self.num_patches = 250
            
            # Training Configuration
            self.batch_size = 15
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 8
            self.nb_epochs = 320
            self.cuda = True

            # Optimizer
            self.lr = 0.001
            self.weight_decay = 0

            # Hyperparameters for the contrastive loss
            self.sigma = 5  # RBF kernel sigma
            self.temperature = 0.1  # Temperature for contrastive loss scaling
            
            # Additional hyperparameters for the prototype loss
            self.num_endmembers = 3  # Number of prototypes
            self.n_bands = 156   # Dimensionality of latent embeddings

            # Augmentations
            self.tf = "crop_only"  # Options: "all_tf", "crop_blur", etc.

            # Model
            self.model = "Autoencoder"  # Options: "DenseNet", "UNet"

            # Paths to the data
            self.data_train = "./Datasets/Samson.mat"  
            # self.data_train = "./Datasets/Urban4.mat"  

            # Input shape (modify based on hyperspectral images)
            self.input_size = (self.n_bands, self.patch_size, self.patch_size)  # Example: (channels, bands, height, width)
            self.label_name = "max abundances"

            # Checkpoint Directory
            self.checkpoint_dir = "./Checkpoint"

        elif self.mode == FINE_TUNING:
            ## Fine-Tuning Configuration (if needed for later classification tasks)
            self.batch_size = 8
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True

            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            # Model and classification task
            self.pretrained_path = "/path/to/model.pth"
            self.num_classes = 2  # Number of classes for classification
            self.model = "DenseNet"  # Options: "DenseNet", "UNet"
            
            
            
            
            