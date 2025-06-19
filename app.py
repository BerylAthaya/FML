import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import os

# 1. ===== SETUP DASAR =====
st.set_page_config(page_title="Deteksi Emosi", layout="centered")

# 2. ===== LOAD MODEL (DENGAN ERROR HANDLING) =====
@st.cache_resource
def init_model():
    try:
        # Pastikan file ada
        if not os.path.exists('emotion_model.onnx'):
            st.error("File model ONNX tidak ditemukan!")
            st.stop()
            
        # Inisialisasi ONNX Runtime
        sess = ort.InferenceSession('emotion_model.onnx',
                                  providers=['CPUExecutionProvider'])
        return sess
    except Exception as e:
        st.error(f"GAGAL MEMUAT MODEL: {str(e)}")
        st.stop()

# 3. ===== DETEKSI WAJAH =====
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces, gray

# 4. ===== PREPROCESSING =====
def preprocess_face(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype(np.float32) / 255.0
    return np.expand_dims(face_roi, axis=(0, -1))  # Shape: (1, 48, 48, 1)

# 5. ===== ANTARMUKA UTAMA =====
model = init_model()
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

st.title("APLIKASI DETEKSI EMOSI")
uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Deteksi wajah
        faces, gray = detect_faces(img)
        
        if len(faces) == 0:
            st.warning("Tidak terdeteksi wajah")
        else:
            for (x, y, w, h) in faces:
                # Preprocessing
                face = preprocess_face(gray[y:y+h, x:x+w])
                
                # Prediksi
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: face})
                pred = outputs[0][0]
                
                # Hasil
                emotion = emotion_labels[np.argmax(pred)]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            st.image(img, channels="BGR", caption="Hasil Deteksi")
            
            # Tampilkan probabilitas
            st.subheader("Kemungkinan Emosi:")
            for label, prob in zip(emotion_labels, pred):
                st.write(f"{label}: {prob:.4f}")
                
    except Exception as e:
        st.error(f"ERROR: {str(e)}")
