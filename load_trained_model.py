# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Define the path to your model directory
model_directory = "D:/My codes/My DL train/Autoencoder/Blood Cell Segmentation"

# Load the trained U-Net model
def load_model(model_directory):
    model_json_path = os.path.join(model_directory, 'model.json')
    model_weights_path = os.path.join(model_directory, 'model_weights.h5')
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    return loaded_model

model = load_model(model_directory)

# Load and preprocess image for model prediction
def load_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, color_mode='grayscale')
    img_resized = img.resize(target_size)
    img_array = img_to_array(img_resized)
    img_array /= 255.
    return img_array, img

# Predict segmentation
def predict_segmentation(image_path, model, target_size=(128, 128)):
    img_array, original_img = load_preprocess_image(image_path, target_size)
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_mask = prediction[0, :, :, 0]
    predicted_mask_resized = Image.fromarray((predicted_mask * 255).astype('uint8'))
    predicted_mask_resized = predicted_mask_resized.resize(original_img.size, Image.NEAREST)
    return original_img, predicted_mask_resized

# Function to plot the comparison for a list of images
def all_images(image_paths, model, base_images_path, base_masks_path):
    plt.figure(figsize=(15, 15))
    
    for i, image_name in enumerate(image_paths):
        full_image_path = os.path.join(base_images_path, image_name)
        mask_path_png = os.path.join(base_masks_path, os.path.splitext(image_name)[0] + '.png')
        mask_path_jpg = os.path.join(base_masks_path, os.path.splitext(image_name)[0] + '.jpg')
        mask_path = mask_path_png if os.path.exists(mask_path_png) else mask_path_jpg
        
        if not os.path.exists(full_image_path):
            print(f"No image found for {full_image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"No mask found for {mask_path}")
            continue

        original_img, predicted_mask = predict_segmentation(full_image_path, model)

        plt.subplot(len(image_paths), 3, 3*i + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title('Original', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.subplot(len(image_paths), 3, 3*i + 2)
        real_mask = load_img(mask_path, color_mode='grayscale')
        plt.imshow(real_mask, cmap='gray')
        plt.title('True Mask', fontsize=16, fontweight='bold')
        plt.axis('off')

        plt.subplot(len(image_paths), 3, 3*i + 3)
        plt.imshow(original_img, cmap='gray')
        plt.imshow(predicted_mask, cmap='viridis', alpha=0.5)
        plt.title('Predicted Mask', fontsize=16, fontweight='bold')
        plt.axis('off')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_images_path = os.path.join(script_dir, 'selected_images.png')
    plt.tight_layout()
    plt.savefig(all_images_path, dpi=300, facecolor='white')

# Base paths for images and masks
base_images_path = "D:/Data for machine learning/Blood Cell Segmentation Dataset/BCCD Dataset with mask/test/original/"
base_masks_path = "D:/Data for machine learning/Blood Cell Segmentation Dataset/BCCD Dataset with mask/test/mask/"

# Example usage
image_paths = [
    "e3ade58d-086c-47fa-9120-76beacb45395.png",
    "e36cb882-c6d0-4467-812e-d18c169a9a47.png",
    "f0ee03d6-c57b-43a0-8812-359330bdb93a.jpg",
    "fe851c88-692d-4199-87e0-d19d9c4eb591.png"
]

all_images(image_paths, model, base_images_path, base_masks_path)