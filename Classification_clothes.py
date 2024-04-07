import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Cargar los datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Definir las etiquetas de clase
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Ver la forma de los datos
print("Forma de los datos de entrenamiento:", x_train.shape)
print("Forma de los datos de prueba:", x_test.shape)

# Visualizar algunos ejemplos de imágenes y sus etiquetas correspondientes
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()
