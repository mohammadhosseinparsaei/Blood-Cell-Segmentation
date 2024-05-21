# Import library
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = "D:\\Data for machine learning\\Blood Cell Segmentation Dataset\\BCCD Dataset with mask"
TARGET_SIZE = (256, 256)

# Paths
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")

# Data loading and preprocessing functions
def find_mask(image_name, mask_dir):
    """Find the corresponding mask file for an image."""
    base_name = os.path.splitext(image_name)[0]
    for ext in ['.png', '.jpg']:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            return mask_path
    return None

def load_image(path):
    """Load and preprocess an image."""
    img = load_img(path, target_size=TARGET_SIZE, color_mode='grayscale')
    image_array = img_to_array(img)
    image_array /= 255.
    return image_array

def load_images_and_masks(dataframe):
    """Load images and masks from DataFrame."""
    images, masks = [], []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        image = load_image(row["image_path"])
        mask = load_image(row["mask_path"])
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)

def prepare_dataframes(train_dir, test_dir):
    """Prepare DataFrames for training and testing datasets."""
    def get_image_and_mask_paths(data_dir):
        image_paths, mask_paths = [], []
        for image_name in os.listdir(os.path.join(data_dir, "original")):
            image_path = os.path.join(data_dir, "original", image_name)
            mask_path = find_mask(image_name, os.path.join(data_dir, "mask"))
            if mask_path:
                image_paths.append(image_path)
                mask_paths.append(mask_path)
        return image_paths, mask_paths

    train_image_paths, train_mask_paths = get_image_and_mask_paths(train_dir)
    test_image_paths, test_mask_paths = get_image_and_mask_paths(test_dir)
    
    train_df = pd.DataFrame({"image_path": train_image_paths, "mask_path": train_mask_paths})
    test_df = pd.DataFrame({"image_path": test_image_paths, "mask_path": test_mask_paths})

    return train_df, test_df

def rotate_images_and_masks(images, masks):
    """Rotate images and masks for data augmentation."""
    rotated_images = []
    rotated_masks = []
    angles = np.arange(90, 271, 90).tolist()

    for img, mask in zip(images, masks):
        for angle in angles:
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_img = cv2.warpAffine(img, M, (cols, rows))
            rotated_mask = cv2.warpAffine(mask, M, (cols, rows))
            rotated_images.append(rotated_img)
            rotated_masks.append(rotated_mask)

    return np.array(rotated_images).reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1), np.array(rotated_masks).reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)

def augment_data(images, masks):
    """Apply data augmentation and shuffle the dataset."""
    rotated_images, rotated_masks = rotate_images_and_masks(images, masks)
    images = np.concatenate((images, rotated_images), axis=0)
    masks = np.concatenate((masks, rotated_masks), axis=0)
    shuffled_indices = np.random.permutation(len(images))
    return images[shuffled_indices], masks[shuffled_indices]

# Metrics
def dice_metric(y_true, y_pred):
    """Calculate Dice coefficient."""
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
    return (2. * intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) + 1.)

def iou_metric(y_true, y_pred):
    """Calculate Intersection over Union (IoU)."""
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection
    return (intersection + 1.) / (union + 1.)

# Callbacks
class MyEarlyStopping(Callback):
    """Custom early stopping callback."""
    def __init__(self, monitor='val_iou_metric', patience=10, verbose=1):
        super(MyEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best_iou = float('-inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_iou = logs.get(self.monitor)
        if current_iou is not None:
            if current_iou > self.best_iou:
                self.best_iou = current_iou
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.verbose > 0:
                        print(f"Epoch {epoch + 1}: early stopping")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0 and self.verbose > 0:
            best_epoch = self.stopped_epoch - self.wait
            print(f"Restored model weights from epoch {best_epoch + 1}")

# Model definition
def define_unet_model(input_shape):
    """Define the U-Net model architecture."""
    inputs = Input(input_shape)

    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def find_best_threshold(y_true, y_pred):
    """Find the best threshold for binary segmentation."""
    best_threshold = 0
    best_iou = 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        thresholded_masks = (y_pred > threshold).astype('float32')
        iou = np.mean([iou_metric(y_true[i], thresholded_masks[i]) for i in range(len(y_true))])
        
        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold
    return best_threshold

# Prepare DataFrames
train_df, test_df = prepare_dataframes(TRAIN_DIR, TEST_DIR)

# Load and preprocess images and masks
train_images, train_masks = load_images_and_masks(train_df)
test_images, test_masks = load_images_and_masks(test_df)

# Split the data into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Augment the training data
train_images, train_masks = augment_data(train_images, train_masks)

# Define and compile the model
input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 1)
model = define_unet_model(input_shape)
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_metric, iou_metric])

# Callbacks for early stopping and model checkpoint
early_stopping = MyEarlyStopping(monitor='val_iou_metric', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_iou_metric', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=100,
    batch_size=2,
    callbacks=[early_stopping, model_checkpoint]
)

# Predict masks for the testing set
predicted_masks = model.predict(test_images)

# Find and apply the best threshold
best_threshold = find_best_threshold(test_masks, predicted_masks)
thresholded_masks = (predicted_masks > best_threshold).astype('float32')

# Calculate evaluation metrics
total_iou, total_dice = 0, 0
max_iou, max_dice = 0, 0
for i in range(len(test_images)):
    iou = iou_metric(test_masks[i], thresholded_masks[i]).numpy()
    dice = dice_metric(test_masks[i], thresholded_masks[i]).numpy()
    total_iou += iou
    total_dice += dice
    if iou > max_iou:
        max_iou = iou
    if dice > max_dice:
        max_dice = dice

average_iou = total_iou / len(test_images)
average_dice = total_dice / len(test_images)

print(f"Average IoU between test data: {average_iou:.4f}")
print(f"Average Dice Coefficient between test data: {average_dice:.4f}")
print(f"Maximum IoU between test data: {max_iou:.4f}")
print(f"Maximum Dice Coefficient between test data: {max_dice:.4f}")

# Plot IoU and Dice coefficients after training
iou_dice_path = os.path.join(SCRIPT_DIR, 'iou_dice_plot.png')
plt.figure(figsize=(8, 8))
epochs = range(1, len(history.history['iou_metric']) + 1)
plt.plot(epochs, history.history['iou_metric'], label='Training IoU')
plt.plot(epochs, history.history['val_iou_metric'], label='Validation IoU')
plt.plot(epochs, history.history['dice_metric'], label='Training Dice')
plt.plot(epochs, history.history['val_dice_metric'], label='Validation Dice')
plt.text(0.5, 1.05, f'Average IoU between test data: {average_iou:.4f}, Average Dice between test data: {average_dice:.4f}\nMaximum IoU between test data: {max_iou:.4f}, Maximum Dice between test data: {max_dice:.4f}', 
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Epoch')
plt.ylabel('IoU / Dice Coefficient')
plt.legend()
plt.tight_layout()
plt.savefig(iou_dice_path, dpi=300, facecolor='white')

# Plot the actual image, its mask, and predicted mask for each selected index
plt.figure(figsize=(8, 16))
for i in range(4):
    idx = np.random.randint(len(test_images))
    image = test_images[idx]
    mask = test_masks[idx]
    pred_mask = model.predict(image[np.newaxis, ...])[0]

    plt.subplot(4, 2, i*2 + 1)
    plt.title("Original", fontsize=16)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.imshow(mask.squeeze(), cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(4, 2, i*2 + 2)
    processed_mask = (pred_mask > best_threshold).astype('float32')
    plt.title(f"Predicted (IoU: {iou_metric(mask, processed_mask).numpy():.4f})", fontsize=16)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.imshow(processed_mask.squeeze(), cmap='jet', alpha=0.5)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'evaluation.png'), dpi=300, facecolor='white')

# Save the trained model
model_json_path = os.path.join(SCRIPT_DIR, "model.json")
model_weights_path = os.path.join(SCRIPT_DIR, "model_weights.h5")
with open(model_json_path, "w") as json_file:
    json_file.write(model.to_json(indent=4))
model.save_weights(model_weights_path)
print("Model architecture and weights saved successfully!")