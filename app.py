import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf
# ... (lanjutkan semua kode app.py yang versi fix tadi ke bawah sini)

# 1. Judul dan Tampilan Web
st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")
st.title("🛡️ Deteksi Penyakit LSD Sapi")
st.write("Aplikasi ini menggunakan arsitektur MobileNetV3 untuk mengidentifikasi gejala Lumpy Skin Disease.")

# 2. Fungsi Memanggil Model
@st.cache_resource
def load_my_model():
    # Pastikan nama ini sama dengan file keras yang kamu download tadi
    model = tf.keras.models.load_model('model_lsd_sapi.keras', compile=False)
    return model

model = load_my_model()

# 3. Daftar Label (Sesuaikan dengan urutan di Colab kamu)
# Di Confusion Matrix kamu: healthy (sehat) dan lumpy skin (LSD)
CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

# 4. Fungsi Prediksi
def predict(image_data, model):
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Preprocessing khusus MobileNetV3
    img_reshape = img_array[np.newaxis, ...]
    img_final = tf.keras.applications.mobilenet_v3.preprocess_input(img_reshape)
    
    prediction = model.predict(img_final)
    result = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return result, confidence

# 5. Fitur Upload Foto
uploaded_file = st.file_uploader("Upload foto tekstur kulit sapi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Foto yang dianalisis', use_container_width=True)
    
    if st.button("Analisis Gambar"):
        with st.spinner('Sedang mendiagnosis...'):
            label, score = predict(image, model)
            
            st.write("---")
            if 'LSD' in label:
                st.error(f"### Hasil: {label}")
                st.warning(f"Tingkat Keyakinan: {score:.2f}%")
                st.write("**Saran:** Segera hubungi dokter hewan dan pisahkan sapi dari kelompoknya.")
            else:
                st.success(f"### Hasil: {label}")
                st.info(f"Tingkat Keyakinan: {score:.2f}%")
