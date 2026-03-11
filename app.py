import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")

st.title("🛡️ Deteksi Penyakit LSD Sapi")
st.write("Aplikasi ini menggunakan arsitektur MobileNetV3 untuk mengidentifikasi gejala Lumpy Skin Disease.")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("model_lsd_sapi.keras", compile=False)
    return model

model = load_my_model()

model = load_my_model()

CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

def predict(image_data, model):

    image = image_data.convert("RGB")
    image = ImageOps.fit(image, (224,224), Image.Resampling.LANCZOS)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

    prediction = model.predict(img_array)

    result = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))*100

    return result, confidence

uploaded_file = st.file_uploader("Upload foto tekstur kulit sapi...", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Foto yang dianalisis", use_container_width=True)

    if st.button("Analisis Gambar"):

        with st.spinner("Sedang mendiagnosis..."):

            label, score = predict(image, model)

            st.write("---")

            if "LSD" in label:
                st.error(f"Hasil: {label}")
                st.warning(f"Tingkat Keyakinan: {score:.2f}%")
                st.write("Saran: Segera hubungi dokter hewan dan pisahkan sapi dari kelompoknya.")
            else:
                st.success(f"Hasil: {label}")
                st.info(f"Tingkat Keyakinan: {score:.2f}%")
