import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2  # OpenCV library for image processing

# Define paths to your dataset folders
train_dir = "./dataset"

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values to [0, 1]
    validation_split=0.2,         # Use 20% of data for validation
    shear_range=0.2,              # Random shearing for data augmentation
    zoom_range=0.2,               # Random zooming
    horizontal_flip=True          # Random horizontal flipping
)

# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),       # Resize images to 150x150 pixels
    batch_size=32,
    class_mode='binary',          # Binary classification
    subset='training'             # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'           # Set as validation data
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    epochs=50
)

# Evaluate on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Load and preprocess a single image
img_path = f"{train_dir}/yes/p1 (30).jpg"  # Example image
img = image.load_img(img_path, target_size=(150, 150))  # Resize image
img_array = image.img_to_array(img) / 255.0  # Normalize
plt.imshow(img_array)
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape

# Predict the class (0 for 'no clot', 1 for 'yes clot')
prediction = model.predict(img_array)
if prediction > 0.5:
    print("Prediction: Yes, Blood clot detected")
else:
    print("Prediction: No, No blood clot")

# Apply Sobel filter to the original image (convert to grayscale first)
img_gray = cv2.cvtColor(img_array[0], cv2.COLOR_RGB2GRAY)  # Convert to grayscale
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)    # Sobel filter in X direction
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)    # Sobel filter in Y direction
sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)            # Combine both X and Y

# Display the Sobel filter result
plt.figure()
plt.subplot(1, 3, 1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 2), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(1, 3, 3), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.show()  

model.save('brain_clot_model.h5')