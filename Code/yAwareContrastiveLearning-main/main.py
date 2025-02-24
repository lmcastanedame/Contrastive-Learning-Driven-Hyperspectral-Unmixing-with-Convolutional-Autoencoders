import numpy as np
from dataset import HyperspectralDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss
from torch.nn import CrossEntropyLoss
from models.autoencoder import Autoencoder
import argparse
from config import Config, PRETRAINING, FINE_TUNING


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly!")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    # Load configuration
    config = Config(mode)

    # Initialize datasets
    if config.mode == mode:
        dataset_train = HyperspectralDataset(config, training=True)
        dataset_val = HyperspectralDataset(config, validation=True)
    else:
        # Placeholder for fine-tuning dataset
        dataset_train = Dataset()
        dataset_val = Dataset()

    # Data loaders
    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers)
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers)

    # Initialize the model
    if config.mode == PRETRAINING:
        if config.model == "Autoencoder":
            net = Autoencoder(config)
        else:
            raise ValueError(f"Unknown model: {config.model}")
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # Initialize the loss function
    if config.mode == PRETRAINING:
        # loss = GeneralizedSupervisedNTXenLoss(
        #     temperature=config.temperature,
        #     kernel='dot',
        #     sigma=config.sigma,
        #     return_logits=True
        # )
        loss = NTXenLoss(temperature=0.1, return_logits=True)
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()

    # Create the training model
    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    # Run pretraining or fine-tuning
    if config.mode == PRETRAINING:
        print("Starting pretraining...")
        model.pretraining()
    else:
        print("Starting fine-tuning...")
        model.fine_tuning()