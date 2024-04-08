import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from PIL import Image
import numpy as np

# Load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image to a 784-element vector
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model using the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test dataset accuracy: {test_accuracy}')




# Load the image
image = Image.open('image.jpg')

# Convert the image to grayscale
image = image.convert('L')

# Resize the image to 28x28 pixels
image = image.resize((28, 28))

# Convert the image to a numpy array
image = np.array(image)

# Normalize the image
image = image / 255.0

# Add a batch dimension to the image
image = np.expand_dims(image, axis=0)

# Make a prediction
prediction = model.predict(image)

# Get the class label with the highest probability
predicted_class = np.argmax(prediction, axis=1)

# Label definitions
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f'Predicted class: {predicted_class[0]}')
print(type(predicted_class[0].item()))
print(class_names[predicted_class[0].item()])

