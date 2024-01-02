import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_prueba = datos["train"], datos["test"]

# Normalizar datos de entrada
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas

# Normalizar los datos de entrenamiento y pruebas con la función
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

# Guardar en cache
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()

# for imagen, etiqueta in datos_entrenamiento.take(1):
#     break
# imagen = imagen.numpy().reshape((28, 28))

# plt.figure()
# plt.imshow(imagen, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28, 1)), 
    tf.keras.layers.Dense(50, activation=tf.nn.relu), 
    tf.keras.layers.Dense(50, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

modelo.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

TAMANO_LOTE = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_prueba = datos_prueba.batch(TAMANO_LOTE)

historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))

