import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

# Load the MNIST dataset for digit classification
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Display the number of training images
num_train_images = len(train_images)

# Display the number of test images
num_test_images = len(test_images)

# Display the shape of the first training image
image_shape = train_images[0].shape

# Display the pixel values of the first training image
first_image_pixels = train_images[0]

# Show the first training image using matplotlib
plt.matshow(first_image_pixels)

# Display the label of the third training image
label_of_third_image = train_labels[2]

# Display the labels of the first five training images
labels_of_first_five_images = train_labels[:5]

# Display the shape of the training dataset
train_dataset_shape = train_images.shape

# Normalize pixel values to be between 0 and 1
train_images_normalized = train_images / 255
test_images_normalized = test_images / 255

# Flatten the images to 1D arrays
flattened_train_images = train_images_normalized.reshape(len(train_images_normalized), 28 * 28)
flattened_test_images = test_images_normalized.reshape(len(test_images_normalized), 28 * 28)

# Display the flattened pixel values of the first training image
flattened_first_image_pixels = flattened_train_images[0]

# Build a simple neural network model for digit classification
model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the flattened training images and labels for 5 epochs
model.fit(flattened_train_images, train_labels, epochs=5)

# Evaluate the model on the flattened test images and labels
evaluation_result = model.evaluate(flattened_test_images, test_labels)

# Display the evaluation result
print("Evaluation Result:", evaluation_result)

# Display the first test image
plt.matshow(test_images[0])

# Display the second test image
plt.matshow(test_images[1])

# Predictions for the test images
predictions = model.predict(flattened_test_images)

# Display the predicted label for the second test image
predicted_label_for_second_image = np.argmax(predictions[1])

# Display the predicted labels for the first five test images
predicted_labels_for_first_five_images = [np.argmax(i) for i in predictions[:5]]

# Display the true labels for the first five test images
true_labels_for_first_five_images = test_labels[:5]

# Create a confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=predicted_labels_for_first_five_images)

# Visualize the confusion matrix using seaborn
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

# Build a more complex neural network model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile and train the new model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(flattened_train_images, train_labels, epochs=5)

# Evaluate the new model on the flattened test images and labels
new_evaluation_result = model.evaluate(flattened_test_images, test_labels)

# Display the new evaluation result
print("New Model Evaluation Result:", new_evaluation_result)

# Visualize the confusion matrix for the new model
new_predictions = model.predict(flattened_test_images)
new_predicted_labels = [np.argmax(i) for i in new_predictions]
new_confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=new_predicted_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(new_confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')

# Build a convolutional neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile and train the convolutional model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
