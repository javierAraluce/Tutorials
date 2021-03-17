#%%
import pandas
import tensorflow as tf
import re
import nltk
import numpy

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Disable eager execution
tf.compat.v1.disable_eager_execution()

nltk.download('stopwords')


dataframe = pandas.read_csv("RNN/data/googleplaystore_user_reviews.csv")
print(dataframe.head())
print(dataframe.tail())

# Antes de preparar los datos para que sean aptos para la entrada de la red 
# neuronal, se eliminarán aquellas filas que contengan NaN.
dataframe = dataframe.dropna()
print(dataframe.head())
# Antes de proceder al procesamiento de los datos, se puede realizar una 
# exploración previa de las frecuencias de los sentimientos mediante un 
# diagrama de barras:
dataframe['Sentiment'].value_counts().plot(kind = 'bar')

dataframe = dataframe[['Translated_Review','Sentiment']]
dataframe.head()

# En principio, tenemos tres clases, de manera que podemos optar por estas dos 
# vías:

# Considerar una clasificación binaria. Eliminaremos del dataframe aquellas 
# muestras que pertenezcan a la categoría Neutral. Convertiremos el vector 
# Sentiment en 0 o 1 según si es, respectivamente, negativo o positivo.
# Considerar la clasificación con los tres tipos. En este caso, deberemos 
# formular una representación one-hot.
# A partir de las columnas que nos interesan, se deben realizar los siguientes 
# pasos para poder unificar todas las frases:

# Poner todas las letras en minúscula.
# Eliminar signos de puntuación, convirtiendo todas las palabras que se 
# encontraban juntas en palabras por separado.
# Eliminar las stop words: Palabras cortas que carecen de significado por sí 
# mismas, como las conjunciones o preposiciones.
# Definimos una función que se encargará de esto, dada la frase sen por entrada:

def preprocess_text(sen):
    # Eliminar símbolos de puntuación y números
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Eliminar carácteres sueltos
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Eliminar espacios excesivos
    sentence = re.sub(r'\s+', ' ', sentence)

    # Convertir a minúscula
    sentence = sentence.lower()
  
    # Eliminar las stopwords.
    words = sentence.split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    
    return ' '.join(filtered_words)

# En la función anterior se está haciendo referencia a la librería re, que 
# viene cargada por defecto en Python. Es una librería dedicada para el 
# tratamiento de expresiones regulares. Por consiguiente, tendremos que aplicar 
# la función anterior a cada elemento de la columna Translated_Review del 
# dataframe.

dataframe['Translated_Review'] = dataframe['Translated_Review'].apply(lambda sen: preprocess_text(sen))
dataframe

# Representación vectorial del texto

# En esta sección se tendrá por objetivo convertir una cadena de texto en un 
# vector, para que pueda ser tratado por la red neuronal. En primer lugar, para 
# poder generar los conjuntos de entrenamiento y validación, separaremos el 
# dataframe en dos variables independientes para poder tratarlas por separado:
def determine_class(label):
  if label == 'Positive':
    return 0
  elif label == 'Neutral':
    return 1
  elif label == 'Negative':
    return 2

# Poner la variable a True si se quieren eliminar los comentarios neutrales.
# En caso contrario, los neutrales se convertirán a negativos
REMOVE_NEUTRAL = False

# Poner la variable a True (siempre que la anterior valga False) para considerar
# las clases 'Negative' y 'Neutral' iguales.
MERGE_NEGATIVE_NEUTRAL = False

if REMOVE_NEUTRAL:
  indexNames = dataframe[dataframe['Sentiment'] == 'Neutral'].index
  dataframe.drop(indexNames , inplace=True)

  y = dataframe['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0).to_numpy()
  #y = tf.one_hot(y, 2)
else:
  if MERGE_NEGATIVE_NEUTRAL:
    y = dataframe['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0).to_numpy()
    #y = tf.one_hot(y, 2)
  else: 
    y = dataframe['Sentiment'].apply(lambda x: determine_class(x)).to_numpy()
    #y = tf.one_hot(y, 3)

X = dataframe['Translated_Review']
y = y.astype(numpy.uint8)



plt.hist(y)
plt.plot()

# A partir de la librería sklearn podemos separar en dos conjuntos disjuntos, 
# conteniendo el 80% de las muestras para el conjunto de entrenamiento y lo 
# restantepara el de validación.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# El siguiente paso es determinar el número máximo de palabras que se usarán 
# (es decir, las  n  primeras palabras más frecuentes) y la longitud máxima de 
# cada vector.
NUMBER_OF_WORDS = 250000
MAX_LEN = 400

tokenizer = Tokenizer(num_words = NUMBER_OF_WORDS)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Después, para aquellas frases que tengan palabras que no sean de las más 
# frecuentes, se realiza la técnica de padding rellenando con ceros aquellas 
# palabras que no sean tan frecuentes y no aparecen.
X_train = pad_sequences(X_train, padding='post', maxlen=MAX_LEN)
X_test = pad_sequences(X_test, padding='post', maxlen=MAX_LEN)

# Generar el modelo de Red Neuronal Recurrente

VOCABULARY_SIZE = NUMBER_OF_WORDS
EMBEDDING_SIZE = 128

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(VOCABULARY_SIZE, 
                                    EMBEDDING_SIZE, 
                                    input_shape=(X_train.shape[1],)))

model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

#En units, se debe especificar cuantas clases tenemos. 
model.add(tf.keras.layers.Dense(units=numpy.unique(y_train).shape[0], 
                                activation='sigmoid'))

model.compile(optimizer='rmsprop', 
                loss='sparse_categorical_crossentropy', 
                metrics=['sparse_categorical_accuracy'])
  
model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=128)

test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))