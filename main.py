import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchvision import transforms
from PIL import Image
import numpy as np


class ColorAutoencoder(nn.Module):
    def __init__(self):
        super(ColorAutoencoder, self).__init__()

        # Encoder for grayscale image
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # Output: 32 x H/2 x W/2
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with a probability of 20%
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x H/4 x W/4
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout with a probability of 30%
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x H/8 x W/8
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout with a probability of 40%
            
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),  # Smaller latent space for reduced complexity
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout with a probability of 50%
        )

        # Decoder to reconstruct the color image
        self.decoder = nn.Sequential(
            nn.Linear(512, 128 * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout with a probability of 50%
            nn.Unflatten(1, (128, 16, 16)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x H/4 x W/4
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout with a probability of 40%
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 32 x H/2 x W/2
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout with a probability of 30%
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # Output: 3 x H x W (RGB)
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the model
@st.cache_resource  # Cache the model to avoid reloading every time
def load_model():
    model = ColorAutoencoder()  # Initialize the model architecture
    model.load_state_dict(torch.load('color_autoencoder.pth', map_location=torch.device('cpu')))  # Load model weights
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the model's expected input size
    transforms.Grayscale(),        # Convert to grayscale
    transforms.ToTensor(),         # Convert to a tensor
])

# Streamlit UI
st.title("Color Autoencoder")
st.write("Upload an image, and the model will attempt to reconstruct it as a cookie.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB for consistency

    # Display the uploaded image
    st.image(input_image, caption='Uploaded Image (Original)', use_column_width=True)

    # Preprocess the image for the model
    preprocessed_image = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        prediction = model(preprocessed_image)

    # Postprocess the model's output
    output_image = prediction.squeeze(0).permute(1, 2, 0).numpy()  # Convert to HxWxC
    output_image = (output_image * 255).astype(np.uint8)  # Scale to [0, 255] for display

    # Convert the output to a PIL image
    output_pil = Image.fromarray(output_image)

    # Display the reconstructed color image
    st.image(output_pil, caption='Reconstructed Image (Color)', use_column_width=True)
