# Importing TensorFlow, a powerful library for machine learning and neural networks
import tensorflow as tf
from tensorflow import keras

# Importing specific classes and functions from TensorFlow for building the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Importing utilities for preprocessing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Importing the warnings library to control warning messages
import warnings

# Configuring the warning filter to ignore all warnings
warnings.filterwarnings('ignore')

# Defining the image dimensions and batch size for training the model
img_height, img_width = 224, 224
batch_size = 50 #tried 30, 16, 100 little difference

# Defining the paths to the training, validation, and test directories
train_dir = 'C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train'
validation_dir = 'C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\validation'
test_dir = 'C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\test'

# location of images
no_tumor = Image.open('C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train\\notumor\\Te-no_0010.jpg')
glioma = Image.open('C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train\\glioma\\Te-gl_0010.jpg')
meningioma = Image.open('C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train\\meningioma\\Te-me_0010.jpg')
pituitary = Image.open('C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train\\pituitary\\Te-pi_0010.jpg')

#plot no_tumor, glioma, meningioma, pituitary with labels on each
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(no_tumor)
plt.title('No Tumor', color = 'blue', fontsize = 12)
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(glioma)
plt.title('Glioma', color = 'blue', fontsize = 12)
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(meningioma)
plt.title('Meningioma', color = 'blue', fontsize = 12)
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(pituitary)
plt.title('Pituitary', color = 'blue', fontsize = 12)
plt.axis('off')
plt.show()

# Creating data generators to rescale images for the training set, between 0 and 1
train_datagen = ImageDataGenerator(rescale=1./255)

# Creating data generators to rescale images for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Creating data generators to rescale images for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Setting up the training data generator to flow from the directory, specifying target size and batch size
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse') #sparse means integer labels instead of one-hot encoding, which converts categorical data into a form that a machine learning algorithm can use, like 0 and 1


# Setting up the validation data generator in a similar manner to the training data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

# Setting up the test data generator, similar to training and validation generators
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size, #look at
    class_mode='sparse')


# Defining the architecture of the Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)), #mess with
    Dropout(0.4),  # Adding a dropout layer to prevent overfitting
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Dropout(0.4),  # Increasing dropout rate for better regularization
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Dropout(0.4),  # Further increasing dropout rate
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding a dropout before the dense layer for regularization
    Dense(4, activation='softmax')  # Output layer with 4 classes
])
#Finding the saliency map of the model



# Compiling the model with Adam optimizer and sparse categorical crossentropy as loss function
#sparse_categorical_crossentropy is needed when your targets are integer values, aka sparse
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generating a summary of the model to view its structure and parameters
model.summary()

# Setting up callbacks for saving the best model and early stopping
# early stopping is used to stop training when the monitored metric has stopped improving 3 = epochs, restore_best_weights means that the model will keep the weights of the best epoch
checkpoint = tf.keras.callbacks.ModelCheckpoint("brain_tumor_classifier_cnn.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Training the model with the training data, validation data, and callbacks
history = model.fit(train_generator, epochs=10,
                    validation_data=validation_generator,
                    callbacks=[checkpoint, early_stopping])

# Evaluating the model's performance on the test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Printing the test accuracy of the model
print('Test accuracy:', test_accuracy)

predictions = model.predict_generator(test_generator, verbose=1)

y_pred = np.argmax(predictions, axis = 1)

label_generator = test_generator.class_indices
classes = list(label_generator.keys())

#output confusion matrix with 4 classes
cm = confusion_matrix(test_generator.classes, y_pred)

tick_labels = np.arange(len(test_generator.class_indices))
# Plotting the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xticks(tick_labels, classes)
plt.xlabel('Predicted Label')
plt.yticks(tick_labels, classes)
plt.ylabel('True Label')
plt.show()

# Plotting the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting the training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print(classification_report(test_generator.classes, y_pred, target_names=classes))


# from keras website, found Grad-CAM to look at class activation heatmaps
last_conv_layer_name = "conv2d_2"

# The local path to our target image
img_path = 'C:\\Users\\mcgee\\OneDrive\\BMI 461\\Presentation 2\\BMI461_CNN_BrainTumor_MRI-20231119T124813Z-001\\BMI461_CNN_BrainTumor_MRI\\archive_MRI\\archive_MRI\\data\\train\\meningioma\\Te-me_0010.jpg'

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Prepare image
img_array = get_img_array(img_path, size=(img_height, img_width))

# Remove last layer's softmax, probabilistic layer, taking raw output scores from previous layer, normalize
model.layers[-1].activation = None

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.show()