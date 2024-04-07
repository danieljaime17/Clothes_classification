import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# this code explores the dataset 


# Load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Show the shapes of the data
print("Shape of training data:", x_train.shape)
print("Shape of training labels:", y_train.shape)
print("Shape of test data:", x_test.shape)
print("Shape of test labels:", y_test.shape)

# Show an example of a training image and its corresponding label
print("Example of a training image:")
print(x_train[0])  # Show the pixel matrix of the first training image
print("Corresponding label:", y_train[0])  # Show the label of the first training image
