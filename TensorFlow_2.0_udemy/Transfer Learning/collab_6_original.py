import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Descomprimir el dataset de Perros vs Gatos
dataset_path = "Transfer Learning/data/cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("Transfer Learning/data")
zip_object.close()

# Configurar las rutas al dataset
dataset_path_new = "Transfer Learning/data//cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")


# Construir el Modelo
# Cargar un modelo pre entrenado (MobileNetV2)
IMG_SHAPE = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model.summary()

# Congelar el modelo base
base_model.trainable = False

# Definir la cabecera personalizada para nuestra red neuronal
base_model.output
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
global_average_layer
prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

# Definir el modelo
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)
model.summary()

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Crear generadores de datos
# Redimensionar imágenes
# Las grandes arquitecturas pre-entrenadas solamente soportan cierto tipo de tamaños de imágenes.
# Por ejemplo: MobileNet (la arquitectura que nosotros usamos) soporta: (96, 96), (128, 128), (160, 160), (192, 192), (224, 224).
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)
train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

# Entrenar el modelo
model.fit_generator(train_generator, epochs=8, validation_data=valid_generator)

# Evaluar el modelo de aprendizaje por transferencia
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
print("Accuracy after transfer learning: {}".format(valid_accuracy))

# Puesta a punto de parámetros
# Un par de cosas:

# NUNCA HAY QUE USAR la puesta a punto (fine tuning) de parámetros en toda la red neuronal: con algunas de las capas superiores (las finales) es más que suficiente suficiente. En la mayoría de casos, son las más especializadas. El objetivo del fine tuning es adaptar esa parte específica de la red neuronal para nuestro nuevo dataset específico.
# Empezar con la puesta a punto DESPUÉS de haber finalizado la fase de aprendizaje por transferencia. Si intentamos hacer el Fine tuning inmediatamente, los gradientes serán muy diferentes entre nuestra cabecera personalizada de la red neuronal y las nuevas capas no congeladas del modelo base.

# Descongelar unas cuantas capas superiores del modelo
base_model.trainable = True
print("Number of layersin the base model: {}".format(len(base_model.layers)))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compilar el modelo para la puesta a punto
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Puesta a punto
model.fit_generator(train_generator,  
                    epochs=5, 
                    validation_data=valid_generator)            

# Evaluar el modelo re calibrado
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
print("Validation accuracy after fine tuning: {}".format(valid_accuracy))