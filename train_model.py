import keras
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# Set your training and validation directories
current_directory = os.getcwd()
train_dir = os.path.join(current_directory, 'train')
log_dir = os.path.join(current_directory, 'logs')

# Define image dimensions and batch size
img_height, img_width = 150, 150
batch_size = 100

# load the training dataset
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

scaled_trained_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# split data into training data batch and validation batch
filenames = os.listdir(train_dir)
labels = [int(filename.startswith("positive")) for filename in filenames]
train_files, validation_files, _, _ = train_test_split(
    filenames, labels, test_size=0.2, stratify=labels, random_state=42
)

# Create data generators for training and validation
train_batches = scaled_trained_data
validation_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Building the feature extraction CNN model
input_shape = (img_height, img_width, 3)
input_layer = keras.layers.Input(shape=input_shape)

conv1 = keras.layers.Conv2D(32, (3, 3), 1, activation='relu', input_shape=input_shape)(input_layer)
pool1 = keras.layers.MaxPooling2D(2, 2)(conv1)

conv2 = keras.layers.Conv2D(64, (3, 3), 1, activation='relu')(pool1)
pool2 = keras.layers.MaxPooling2D(2, 2)(conv2)

conv3 = keras.layers.Conv2D(128, (3, 3), 1, activation='relu')(pool2)
pool3 = keras.layers.MaxPooling2D(2, 2)(conv3)

flatten = keras.layers.Flatten()(pool3)
extracted_features = keras.layers.Dense(256, activation='relu')(flatten)

# Add an embedding layer
embedding_layer = keras.layers.Dense(128, activation='relu')(extracted_features)

# Output layer for similarity prediction
output_layer = keras.layers.Dense(1, activation='sigmoid')(embedding_layer)

feature_extraction_model = keras.models.Model(inputs=input_layer, outputs=output_layer)

feature_extraction_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

print(feature_extraction_model.summary())

# train the model
tensorboard_callback = keras.callbacks.TensorBoard(log_dir)

history = feature_extraction_model.fit(
    train_batches,
    epochs=20,
    validation_data=validation_batches,
    callbacks=[tensorboard_callback]
)

# Save the trained model
feature_extraction_model.save(os.path.join("saved_models", "footprint_auth_model.h5"))

# Plot performance
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].plot(history.history["loss"], color="teal", label="loss")
axes[0].set_title("Loss", fontsize=15)
axes[0].legend(loc="upper right")

axes[1].plot(history.history["accuracy"], color="teal", label="accuracy")
axes[1].set_title("Accuracy", fontsize=15)
axes[1].legend(loc="upper right")

plt.show()