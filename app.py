import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import os

# Konfigurasi
st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="😊")

@st.cache_resource
def load_model():
    try:
        return ort.InferenceSession('emotion_model.onnx')
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

@st.cache_resource
def load_cascade():
    try:
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Gagal memuat Haar Cascade: {str(e)}")
        st.stop()

# Periksa file
if not all(os.path.exists(f) for f in ['emotion_model.onnx', 'haarcascade_frontalface_default.xml']):
    st.error("File model atau Haar Cascade tidak ditemukan!")
    st.stop()

# Muat model dan cascade
model = load_model()
face_cascade = load_cascade()

# Label emosi
emotions = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# UI
st.title("Deteksi Ekspresi Wajah")
uploaded_file = st.file_uploader("Pilih gambar wajah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Format gambar tidak didukung")
            st.stop()
            
        # Deteksi wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            st.warning("Wajah tidak terdeteksi")
        else:
            for (x, y, w, h) in faces:
                # Preprocessing
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=(0, -1))
                
                # Prediksi
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: face})
                pred = outputs[0][0]
                
                # Hasil
                emotion = emotions[np.argmax(pred)]
                confidence = np.max(pred)
                
                # Gambar bounding box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Tampilkan hasil
            st.image(img, channels="BGR", use_column_width=True)
            
            # Tampilkan detail
            st.subheader("Detail Prediksi:")
            for i, (emotion, prob) in enumerate(zip(emotions, pred)):
                st.write(f"{emotion}: {prob:.4f}")
                st.progress(float(prob))
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
