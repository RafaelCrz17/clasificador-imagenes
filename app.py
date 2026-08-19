import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- Config ----------
st.set_page_config(page_title="Clasificador CIFAR-10", page_icon="🤖")
CLASES = ['avion', 'auto', 'pajaro', 'gato', 'ciervo',
          'perro', 'rana', 'caballo', 'barco', 'camion']

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_cifar10.h5')

model = cargar_modelo()

# ---------- Interfaz ----------
st.title("🔍 Clasificador de Imágenes - CIFAR-10")
st.caption("Desarrollado por: JOSE RAFAEL CRUZ RODRIGUEZ") 

opcion = st.radio("Elige una opción:", ["Subir imagen", "Tomar foto"])

imagen = None
if opcion == "Subir imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo:
        imagen = Image.open(archivo)
else:
    foto = st.camera_input("Toma una foto")
    if foto:
        imagen = Image.open(foto)

if imagen is not None:
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    # Preprocesar
    img = imagen.convert('RGB').resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir
    pred = model.predict(img_array)
    clase = CLASES[np.argmax(pred)]
    confianza = float(np.max(pred))

    st.subheader(f"Predicción: **{clase}**")
    st.write(f"Confianza: {confianza:.2%}")
    st.progress(confianza)
