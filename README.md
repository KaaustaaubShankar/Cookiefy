# Cookie Autoencoder

A deep learning model that generates cookies from images through an autoencoder, with ongoing work to preserve more original image features during reconstruction.

## Features

- Converts images into cookies
- Interactive web interface using Streamlit
- Support for JPG and PNG image formats

## Technical Architecture

Convolutional autoencoder with:
- **Encoder**: Compresses grayscale input into latent space (512 dimensions)
- **Decoder**: Reconstructs color image from latent representation
- Uses batch normalization, progressive dropout (20-50%), and residual connections

## Requirements
