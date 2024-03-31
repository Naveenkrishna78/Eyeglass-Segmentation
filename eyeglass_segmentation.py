import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Define the U-Net architecture for segmentation
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Add more convolutional and pooling layers as needed

    up6 = Conv2D(32, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(pool1))
    up6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up6)
    up6 = Conv2D(2, (3, 3), activation='relu', padding='same')(up6)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up6)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

# Preprocess data and load images
def load_images(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, filename))
            img = cv2.resize(img, (256, 256))  # Resize images to match input_size
            images.append(img)
    return np.array(images)

# Perform segmentation using the trained model
def segment_images(model, images):
    segmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        segmented_img = model.predict(img)
        segmented_images.append(segmented_img)
    return np.array(segmented_images)

# Example usage
data_dir = '/path/to/your/dataset'
output_dir = '/path/to/save/segmented/images'

# Load and preprocess images
images = load_images(data_dir)

# Create and compile the segmentation model
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model (replace placeholders with actual training data and labels)
# model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Perform segmentation on test images
segmented_images = segment_images(model, images)

# Save segmented images
for i, seg_img in enumerate(segmented_images):
    cv2.imwrite(os.path.join(output_dir, f'segmented_{i}.jpg'), seg_img)

print("Segmentation completed and images saved.")
