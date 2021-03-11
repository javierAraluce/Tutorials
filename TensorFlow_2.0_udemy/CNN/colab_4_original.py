import tensorflow as tf

from tensorflow.keras.datasets import cifar10

# Pre procesado de datos
# Cargar el dataset Cifar10

# Configurar el nombre de las clases del dataset
class_names = ['avión', 'coche', 'pájaro', 'gato', 'ciervo', 
                'perro', 'rana', 'caballo', 'barco', 'camión']
# Cargar el dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalización de las imágenes
X_train.shape
X_train = X_train / 255.0
X_test = X_test / 255.0
y_test[10]

# Construir una red neuronal convolucional
model = tf.keras.models.Sequential()


# Añadir la primera capa de convolución
# Hyper parámetros de la capa de la RNC:

# Filtros: 32
# Tamaño del kernel: 3
# padding: same
# Función de Activación: relu
# input_shape: (32, 32, 3)
model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, padding="same", activation="relu", 
        input_shape=[32, 32, 3]))


# Añadir una segunda capa convolucional y la capa de max-pooling
# Hyper parámetros de la capa de la RNC:

# Filtros: 32
# Tamaño del kernel: 3
# padding: same
# Función de Activación: relu
# Hyper parámetros de la capa de MaxPool:

# pool_size: 2
# strides: 2
# padding: valid
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# Añadir la tercera capa convolucional
# Hyper parámetros de la capa de la RNC:

# Filtros: 64
# Tamaño del kernel: 3
# padding: same
# Función de Activación: relu
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

# Añadir la cuarta capa convolucional y la capa de max-pooling
# Hyper parámetros de la capa de la RNC:

# Filtros: 64
# Tamaño del kernel: 3
# padding: same
# Función de Activación: relu
# Hyper parámetros de la capa de la MaxPool:

# pool_size: 2
# strides: 2
# padding: valid

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Añadir la capa de flattening
model.add(tf.keras.layers.Flatten())

# Añadir la primera capa fully-connected
# Hyper parámetros de la capa totalmente conectada:

# units/neurons: 128
# activation: relu
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Añadir la capa de salida
# Hyper parámetros de la capa totalmente conectada:

# units/neurons: 10 (number of classes)
# activation: softmax

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()


# Compilar el modelo
# sparse_categorical_accuracy
# sparse_categorical_accuracy comprueba si el valor verdadero maximal coincide con el índice maximal del valor de la predicción.

# https://stackoverflow.com/questions/44477489/keras-difference-between-categorical-accuracy-and-sparse-categorical-accuracy
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5)


# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

