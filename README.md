# Text-to-Image
Input is taken as preprocessed images with embeddings in h5py format and we generate images of flower that match the description present in embeddings
# GAN Training for Text-to-Image Generation

## Overview

This project implements a Generative Adversarial Network (GAN) for generating images from text descriptions. The model is trained on a dataset of flower images and corresponding text descriptions. The GAN consists of two main components:

- **Generator**: Takes noise and text embeddings as input and generates images.
- **Discriminator**: Evaluates the authenticity of the generated images and matches them with the text descriptions.

## Dataset

The input dataset should consist of preprocessed images and their corresponding text embeddings stored in an HDF5 file format (`.h5`).

## Algorithm

The following steps are performed during training:

1. **Encode Matching Text**: Encode the matching text description for each image.
2. **Encode Mismatching Text**: Encode a mismatching text description.
3. **Generate Noise**: Sample random noise from a Gaussian distribution.
4. **Generate Fake Images**: Use the generator to create fake images from the noise and matching text embeddings.
5. **Compute Discriminator Scores**:
   - Real image with correct text.
   - Real image with incorrect text.
   - Fake image with correct text.
6. **Compute Discriminator Loss**: Calculate the loss for the discriminator.
7. **Update Discriminator Parameters**: Adjust the discriminator's weights.
8. **Compute Generator Loss**: Calculate the loss for the generator.
9. **Update Generator Parameters**: Adjust the generator's weights.

## Requirements

- Python 3.x
- TensorFlow or PyTorch
- h5py

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/vanibalani/Text-to-Image.git
    ```
2. Navigate to the project directory:
    ```bash
    cd text-to-image-gan
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess the Dataset**: Ensure your dataset is in the required format (images and text embeddings in `.h5` format).

2. **Train the Model**:
    ```python
    # Example training script
    from train import train_gan

    train_gan(dataset_path='path/to/dataset.h5', batch_size=64, learning_rate=0.0002)
    ```

3. **Generate Images**:
    ```python
    # Example image generation script
    from generate import generate_images

    generate_images(model_path='path/to/trained_model.pth', text_embedding='text embedding here')
    ```

## Example

Here is an example of how to train the model and generate images:

```python
# Train the model
from train import train_gan

train_gan(dataset_path='data/flowers.h5', batch_size=64, learning_rate=0.0002)

# Generate images
from generate import generate_images

generate_images(model_path='models/generator.pth', text_embedding='A pink flower with green leaves.')
