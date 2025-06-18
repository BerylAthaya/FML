import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import os

# Konfigurasi
st.set_page_config(page_title="Deteksi Ekspresi Wajah ONNX", page_icon="😊")

# Inisialisasi ONNX Runtime
@st.cache_resource
def load_onnx_model():
    try:
        sess = ort.InferenceSession('emotion_model.onnx')
        return sess
    except Exception as e:
        st.error(f"Gagal memuat model ONNX: {str(e)}")
        st.stop()

# Load Haar Cascade
@st.cache_resource
def load_haar_cascade():
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Gagal memuat Haar Cascade: {str(e)}")
        st.stop()

# Periksa file yang diperlukan
if not os.path.exists('emotion_model.onnx'):
    st.error("File model ONNX tidak ditemukan!")
    st.stop()

if not os.path.exists('haarcascade_frontalface_default.xml'):
    st.error("File Haar Cascade tidak ditemukan!")
    st.stop()

# Muat model dan cascade
ort_session = load_onnx_model()
face_cascade = load_haar_cascade()

# Label emosi
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Antarmuka Streamlit
st.title("Deteksi Ekspresi Wajah dengan ONNX")
uploaded_file = st.file_uploader("Upload gambar wajah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Baca dan proses gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Gagal memproses gambar")
            st.stop()
        
        # Konversi ke grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
        
        if len(faces) == 0:
            st.warning("Tidak terdeteksi wajah")
        else:
            for (x, y, w, h) in faces:
                # Ekstrak ROI wajah
                face_roi = gray_img[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype(np.float32) / 255.0
                face_roi = np.expand_dims(face_roi, axis=(0, -1))  # Shape: (1, 48, 48, 1)
                
                # Prediksi dengan ONNX Runtime
                input_name = ort_session.get_inputs()[0].name
                output_name = ort_session.get_outputs()[0].name
                pred = ort_session.run([output_name], {input_name: face_roi})[0][0]
                
                emotion_idx = np.argmax(pred)
                emotion = emotion_labels[emotion_idx]
                confidence = pred[emotion_idx]
                
                # Gambar bounding box
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            st.image(image, channels="BGR", use_column_width=True)
            
            # Tampilkan detail prediksi
            st.subheader("Probabilitas Emosi:")
            for label, prob in zip(emotion_labels, pred):
                st.progress(float(prob), text=f"{label}: {prob:.4f}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
