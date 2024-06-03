import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Set the working directory
os.chdir('/Users/parva4/Documents/denoise')

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model.h5')

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L') 
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def save_image(img_array, save_path):
    img_array = img_array.squeeze()  # Remove batch dimension and channel dimension
    img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8 type and scale to [0, 255]
    img = Image.fromarray(img_array, mode='L')  # Convert to PIL Image
    img.save(save_path)

# Path to the noisy image
img_path = '/Users/parva4/Desktop/noisy - 32.jpg'

# Preprocess the image
x_test = load_and_preprocess_image(img_path)

# Make predictions on the test image
predictions = autoencoder.predict(x_test)

# Save the reconstructed image
save_image(predictions, 'sign_pred.jpg')

print('Reconstructed image saved as sign_pred.jpg')
