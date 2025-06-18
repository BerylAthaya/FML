import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('emotion_model.h5')
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

st.title("Deteksi Ekspresi Wajah")

uploaded_file = st.file_uploader("Upload gambar wajah...", type=["jpg", "png"])
if uploaded_file is not None:
    # Perbaikan di sini:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Pakai flags
    
    # Konversi ke grayscale untuk model
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48)) / 255.0
    gray = np.expand_dims(gray, axis=[0, -1])
    
    pred = model.predict(gray)[0]
    emotion = emotion_labels[np.argmax(pred)]
    
    st.image(image, caption=f"Hasil: {emotion}", width=300)
