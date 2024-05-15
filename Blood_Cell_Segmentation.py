import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Script path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set dataset paths
root_dir = "D:\\Data for machine learning\\Blood Cell Segmentation Dataset\\BCCD Dataset with mask"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

# Initialize lists for storing paths and labels
train_image_paths, train_mask_paths = [], []
test_image_paths, test_mask_paths = [], []

# Function to find corresponding mask file
def find_mask(image_name, mask_dir):
    base_name = os.path.splitext(image_name)[0]
    for ext in ['.png', '.jpg']:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            return mask_path
    return None

# Load training images and masks
for image_name in os.listdir(os.path.join(train_dir, "original")):
    image_path = os.path.join(train_dir, "original", image_name)
    mask_path = find_mask(image_name, os.path.join(train_dir, "mask"))
    if mask_path:
        train_image_paths.append(image_path)
        train_mask_paths.append(mask_path)

# Load test images and masks
for image_name in os.listdir(os.path.join(test_dir, "original")):
    image_path = os.path.join(test_dir, "original", image_name)
    mask_path = find_mask(image_name, os.path.join(test_dir, "mask"))
    if mask_path:
        test_image_paths.append(image_path)
        test_mask_paths.append(mask_path)

# Create DataFrames to organize paths
train_df = pd.DataFrame({"image_path": train_image_paths, "mask_path": train_mask_paths})
test_df = pd.DataFrame({"image_path": test_image_paths, "mask_path": test_mask_paths})

# Function to load and preprocess images
def load_image(path):
    img = load_img(path, target_size=(128, 128), color_mode='grayscale')
    image_array = img_to_array(img)
    image_array /= 255.
    return image_array

# Load all images and masks
def load_data(df):
    images, masks = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = load_image(row["image_path"])
        mask = load_image(row["mask_path"])
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)

x_train, y_train = load_data(train_df)
x_test, y_test = load_data(test_df)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=4)

# Define U-Net model architecture
def unet_model():
    inputs = Input((128, 128, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate the model
unet = unet_model()

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = unet.fit(x_train, y_train,
                    batch_size=16,
                    epochs=100,
                    verbose=1,
                    shuffle=True,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping])

# Function to calculate the IoU and Dice coefficient
def calculate_iou_and_dice(y_true, y_pred):
    """
    Calculates Intersection over Union (IoU) and Dice coefficient for binary masks.
    
    Parameters:
    y_true (numpy array): The ground truth binary mask of shape (height, width).
    y_pred (numpy array): The predicted binary mask of shape (height, width).
    
    Returns:
    tuple: A tuple containing IoU and Dice coefficient.
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate intersection
    intersection = np.sum(y_true * y_pred)

    # Calculate union
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    # Calculate IoU
    iou = intersection / union

    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))

    return iou, dice


# Function to plot images with masks
def plot_images_with_masks(images, masks, predicted_masks, num_images=4):
    plt.figure(figsize=(15, 4*num_images))
    for i in range(num_images):
        plt.subplot(num_images, 3, 3*i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(num_images, 3, 3*i + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(num_images, 3, 3*i + 3)
        plt.imshow(images[i], cmap='gray')
        plt.imshow(predicted_masks[i].squeeze(), alpha=0.5, cmap='viridis')
        plt.title('Predicted Segmentation')
        plt.axis('off')

    plt.tight_layout()
    evaluation_path = os.path.join(script_dir, 'evaluation.png')
    plt.savefig(evaluation_path, dpi=300, facecolor='white')

# Select four random images from the test set
random_indices = np.random.choice(len(x_test), size=4, replace=False)
sample_images = x_test[random_indices]
sample_masks = y_test[random_indices]

# Predict masks for the selected images
predicted_masks = unet.predict(sample_images)

# Plot the images with true and predicted masks
plot_images_with_masks(sample_images, sample_masks, predicted_masks)

# Predict masks for the entire test set
predicted_masks = unet.predict(x_test)

total_iou = 0
total_dice = 0

# Iterate through the test set
for i in range(len(x_test)):
    iou, dice = calculate_iou_and_dice(y_test[i], predicted_masks[i])
    total_iou += iou
    total_dice += dice

# Calculate the average IoU and Dice coefficient over the test set
average_iou = total_iou / len(x_test)
average_dice = total_dice / len(x_test)

print(f"Average IoU: {average_iou}")
print(f"Average Dice coefficient: {average_dice}")

# Plot training history
acc_loss_path = os.path.join(script_dir, 'accuracy_loss_plot.png')
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.title('Training and Validation Accuracy / Loss\n'
          f'Average IoU: {average_iou:.4f}, Average Dice Coefficient: {average_dice:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig(acc_loss_path, dpi=300, facecolor='white')

# Saving the Trained Model
model_json = unet.to_json()
model_json_path = os.path.join(script_dir, "model.json")
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

model_weights_path = os.path.join(script_dir, "model_weights.h5")
unet.save_weights(model_weights_path)

print("Model architecture and weights saved successfully to the script's directory!")