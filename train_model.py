import keras
import os
from matplotlib import pyplot as plt


# Set your training and validation directories
current_directory = os.getcwd()
train_dir = os.path.join(current_directory, 'train')
log_dir = os.path.join(current_directory, 'logs')

# Define image dimensions and batch size
img_height, img_width = 150, 150
batch_size = 100

# load the training dataset
train_datagen = keras.utils.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
)

# scale data into smaller size
scaled_trained_data = train_datagen.map(lambda x,y: (x/255, y))

# split data into training data batch and validation batch
training_size = int(len(scaled_trained_data)*.7)
validation_size = int(len(scaled_trained_data)*.2)+1
testing_size = int(len(scaled_trained_data)*.1)+1

training_batches = scaled_trained_data.take(training_size)
validation_batches = scaled_trained_data.skip(training_size).take(validation_size)
test_batches = scaled_trained_data.skip(training_size + validation_size).take(testing_size)

# building the feature extraction CNN model
input_shape = (img_height, img_width, 3)
input_layer = keras.layers.Input(shape=input_shape)

conv1 = keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=input_shape)(input_layer)
pool1 = keras.layers.MaxPooling2D(2, 2)(conv1)

conv2 = keras.layers.Conv2D(32, (3, 3), 1, activation='relu')(pool1)
pool2 = keras.layers.MaxPooling2D(2, 2)(conv2)

conv3 = keras.layers.Conv2D(16, (3, 3), 1, activation='relu')(pool2)
pool3 = keras.layers.MaxPooling2D(2, 2)(conv3)

flatten = keras.layers.Flatten()(pool3)
extracted_features = keras.layers.Dense(150, activation='relu')(flatten)

feature_extraction_model = keras.models.Model(inputs=input_layer, outputs=extracted_features)

feature_extraction_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

print(feature_extraction_model.summary())

# train the model
tensorboard_callback = keras.callbacks.TensorBoard(log_dir)

history = feature_extraction_model.fit(
    training_batches,
    epochs=10,
    validation_data=validation_batches,
    callbacks=[tensorboard_callback]
)

# plot performance
fig = plt.figure()
plt.plot(history.history["loss"], color="teal", label="loss")
plt.plot(history.history["val_loss"], color="orange", label="val_loss")
fig.suptitle("Loss", fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history["accuracy"], color="teal", label="accuracy")
plt.plot(history.history["val_accuracy"], color="orange", label="val_accuracy")
fig.suptitle("Accuracy", fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Save the trained model
feature_extraction_model.save(os.path.join("saved_models", "footprint_auth_model.h5"))
