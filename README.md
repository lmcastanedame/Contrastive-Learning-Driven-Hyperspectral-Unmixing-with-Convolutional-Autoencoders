# 🛰 Contrastive Learning-Driven Hyperspectral Unmixing with Convolutional Autoencoders

## 📌 Overview

This repository contains the implementation of a contrastive learning-based approach for hyperspectral unmixing using convolutional autoencoders. The objective is to decompose a hyperspectral image into its constituent endmembers and their corresponding abundances, leveraging contrastive learning to enhance feature extraction.

## 🔍 Problem Statement

Hyperspectral unmixing aims to recover pure spectral signatures (endmembers) and their proportions (abundances) from mixed hyperspectral pixels. Traditional unmixing methods rely on linear or nonlinear models, whereas deep learning offers a data-driven approach to extract spectral-spatial representations.

✨ Proposed Approach
- A convolutional autoencoder (CAE) is used to learn a latent representation of hyperspectral patches.
- Contrastive learning is applied to improve the discriminability of feature representations.
- Data augmentations (e.g., cropping, jittering, flipping) are used to generate positive and negative pairs for contrastive training.
- Reconstruction loss ensures that the extracted features retain meaningful spectral information.

## 🚀 Prerequisites

Ensure you have Python 3.8+ installed. Clone this repository:

git clone https://github.com/lmcastanedame/Contrastive-Learning-Driven-Hyperspectral-Unmixing.git
cd Contrastive-Learning-Driven-Hyperspectral-Unmixing

## 📚 References
- A Simple Framework for Contrastive Learning of Visual Representations
- Big Self-Supervised Models are Strong Semi-Supervised Learners
- Contrastive Learning for Blind Hyperspectral Unmixing (CLHU)

## 👩‍💻 Author

Developed by Manuela Castañeda
If you find this repository useful, feel free to ⭐ it!
