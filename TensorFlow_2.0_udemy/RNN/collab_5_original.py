import tensorflow as tf
from tensorflow.keras.datasets import imdb

print (tf.__version__)
# Pre procesado de datos
# Configurar parámetros del dataset
number_of_words = 20000
max_len = 100 # Tecnica de padding, todas tienen 100 palabras

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Carga del dataset de IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Cortar secuencias de texto de la misma longitud
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Configurar parámetros de la capa de Embedding
vocab_size = number_of_words
print(vocab_size)
embed_size = 128

# Construir la Red Neuronal Recurrente
# Definir el modelo

model = tf.keras.Sequential()

# Añadir la capa de embedding
model.add(tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(X_train.shape[1],)))

# Añadir la capa de LSTM
# unidades: 128
# función de activación: tanh
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

# Añadir la capa totalmente conectada de salida
# unidades: 1
# función de activación: sigmoid
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Compilar el modelo
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Entrenar el modelo
model.fit(X_train, y_train, epochs=3, batch_size=128)

# Entrenar el modelo
test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))