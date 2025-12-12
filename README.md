# Drug Molecule Generation with VAE

This project implements a Variational Autoencoder (VAE) for generating valid drug-like molecules using the ZINC dataset. It leverages Relational Graph Convolutional Networks (R-GCN) for the encoder and a dense network for the decoder, capable of transforming SMILES strings into molecular graphs and generating new molecules from the latent space.

## Project Overview

The core of this project is a generative model that learns to represent molecules in a continuous latent space. This allows for:
1.  **Molecule Encoding**: Converting discrete molecular structures (graphs) into continuous vectors.
2.  **Molecule Generation**: Sampling from the latent space to generate novel molecular structures.
3.  **Property Prediction**: Predicting molecular properties directly from the latent representation.

## Dataset

The project uses the **ZINC** dataset (specifically `250k_rndm_zinc_drugs_clean_3.csv`), a free database of commercially available compounds for virtual screening.
The dataset includes:
-   **SMILES**: Simplified Molecular Input Line Entry System representation of molecules.
-   **logP**: Water–octanal partition coefficient.
-   **SAS**: Synthetic Accessibility Score.
-   **QED**: Qualitative Estimate of Drug-likeness.

## Architecture

### 1. Encoder (R-GCN)
The encoder takes the molecular graph (adjacency matrix and feature matrix) as input.
-   **Graph Convolution**: Uses `RelationalGraphConvLayer` to aggregate information from neighboring atoms, respecting bond types.
-   **Dense Layers**: Flattens the graph representation and passes it through dense layers.
-   **Latent Space**: Outputs `z_mean` and `z_log_var` to define the variational distribution.

### 2. Decoder
The decoder reconstructs the molecular graph from the latent vector `z`.
-   **Dense Layers**: Expands the latent vector.
-   **Reshape & Softmax**: Outputs reconstructed adjacency (bond types) and feature (atom types) tensors using Softmax activation to ensure valid probability distributions.

### 3. VAE Loss Function
The model optimizes a composite loss function:
-   **Reconstruction Loss**: Categorical crossentropy for both adjacency and feature matrices.
-   **KL Divergence**: Regularizes the latent space to be close to a standard normal distribution.
-   **Property Prediction Loss**: Binary crossentropy for predicting molecular properties (e.g., QED) from the latent space.
-   **Gradient Penalty**: Enforces 1-Lipschitz continuity to improve training stability (similar to WGAN-GP).

## Hyperparameters

-   **Batch Size**: 32
-   **Epochs**: 10
-   **Learning Rate**: 5e-4
-   **Max Atoms**: 120
-   **Latent Dimension**: 435
-   **Atom Dimension**: 11 (Based on dataset character set)
-   **Bond Dimension**: 5 (Single, Double, Triple, Aromatic, None)

## Prerequisites

-   Python 3.7+
-   TensorFlow
-   RDKit
-   Pandas
-   NumPy
-   Matplotlib

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the dataset file `250k_rndm_zinc_drugs_clean_3.csv` is located in the expected directory (default: `../input/zinc250k/`). *Note: You may need to adjust the path in the notebook if running locally.*
2.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook drug-molecule-generation-with-vae.ipynb
    ```
3.  The notebook includes steps for:
    -   Data loading and preprocessing (SMILES to Graph).
    -   Model building (Encoder, Decoder, VAE).
    -   Training loop with custom `train_step`.
    -   Inferencing to generate new molecules.
