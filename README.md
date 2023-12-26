# Footprint Authentication Model

This repository contains the code for a Convolutional Neural Network (CNN) model that can be used for footprint authentication. The model is trained on a dataset of footprint images and can be used to identify individuals based on their footprints.

## Prerequisites

To run the code in this repository, you will need the following:

- Python 3.6 or later
- TensorFlow 2.0 or later
- Keras 2.3 or later
- ImageDataGenerator from Keras

## Getting Started

To get started, clone this repository to your local machine:

```
git clone https://github.com/innextinit/footprint-authentication-model.git
```

Once you have cloned the repository, you can install the required dependencies by running the following command in the terminal:

```
yarn install
```

## Training the Model

The model is trained on a dataset of footprint images. The dataset is divided into two parts: a training set and a validation set. The training set is used to train the model, while the validation set is used to evaluate the performance of the model.

Get scanned footprint images and move it into the `train/class_1` folder.

After getting the scanned footprint images, to train the model, run the following command in the terminal:

```
yarn train
```

This command will train the model for 20 epochs. The model will be saved to the `saved_models` directory after training.

## Using the Model

The trained model can be used to identify individuals based on their footprints. To use the model, run the project with in dev mode with this command

```
yarn dev
```

The project would be running on https://127.0.0.1:5000. This is have two input fields and two button. The first one is to register a new users footprint. And the second to get authenticate a user, provide the scanned image footprint to the two input field.

## Conclusion

This repository contains the code for a CNN model that can be used for footprint authentication. The model is trained on a dataset of footprint images and can be used to identify individuals based on their footprints.
