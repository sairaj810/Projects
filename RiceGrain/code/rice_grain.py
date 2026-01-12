import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set image size and batch size
img_size = (128, 128)
batch_size = 32
num_classes = 5  # 5 rice varieties

data_dir = "rice_grain_dataset/"  # Change this to your dataset location

# Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 split for train-validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir + 'train/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir + 'train/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model
epochs = 5  # Adjust epochs as needed
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Save the model
model.save("rice_grain_cnn.h5")

# Load the saved model for testing
model = load_model("rice_grain_cnn.h5")

# Evaluate Model on Test Dataset
test_generator = train_datagen.flow_from_directory(
    data_dir + 'test/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

# ==============================
# Single Image Prediction (Commented Out)
# ==============================

# from tensorflow.keras.preprocessing import image
# import numpy as np

# def predict_rice(image_path, model):
#     img = image.load_img(image_path, target_size=img_size)
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction)
    
#     class_names = list(train_generator.class_indices.keys())  # Get class labels
#     print(f"Predicted Variety: {class_names[predicted_class]}")

# Example Usage
# predict_rice("path_to_your_test_image.jpg", model)
