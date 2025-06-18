import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Judul Aplikasi
st.title("😊 Deteksi Ekspresi Wajah dengan Haar Cascade")
st.markdown("""
<style>
.st-emotion-cache-1kyxreq.e115fcil2 {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Load Model dan Haar Cascade
@st.cache_resource
def load_models():
    # Load model ONNX
    ort_session = ort.InferenceSession("emotion_model.onnx")
    
    # Load Haar Cascade
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    return ort_session, face_cascade

ort_session, face_cascade = load_models()
input_name = ort_session.get_inputs()[0].name

# Label Emosi
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Fungsi Deteksi Wajah + Prediksi
def detect_and_predict(image):
    # Konversi ke grayscale untuk Haar Cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    
    results = []
    for (x, y, w, h) in faces:
        # Crop wajah
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess untuk model ONNX
        face_processed = cv2.resize(face_roi, (48, 48))
        face_processed = face_processed / 255.0
        face_processed = np.expand_dims(face_processed, axis=(0, -1)).astype(np.float32)
        
        # Prediksi emosi
        outputs = ort_session.run(None, {input_name: face_processed})
        emotion_idx = np.argmax(outputs[0])
        confidence = np.max(outputs[0])
        
        # Gambar bounding box dan label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 
                   f"{emotion_labels[emotion_idx]} ({confidence*100:.1f}%)", 
                   (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (0, 255, 0), 2)
        
        results.append({
            "bbox": (x, y, w, h),
            "emotion": emotion_labels[emotion_idx],
            "confidence": confidence
        })
    
    return image, results

# Pilihan Input
option = st.radio("Pilih Mode Input:", ("Upload Gambar", "Gunakan Webcam"), horizontal=True)

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar wajah...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(image, caption="Gambar Original", use_column_width=True)
        
        # Proses deteksi
        processed_img, results = detect_and_predict(image.copy())
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Hasil Deteksi", use_column_width=True)
        with col2:
            if results:
                st.subheader("Detail Hasil")
                for i, res in enumerate(results):
                    st.write(f"**Wajah {i+1}**:")
                    st.write(f"- Emosi: {res['emotion']}")
                    st.write(f"- Akurasi: {res['confidence']*100:.2f}%")
            else:
                st.warning("Tidak terdeteksi wajah!")

else:
    st.write("Tekan tombol di bawah untuk mulai deteksi via webcam")
    run_webcam = st.checkbox("Jalankan Webcam")
    
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses webcam")
            break
        
        # Flip horizontal untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Deteksi wajah dan emosi
        processed_frame, _ = detect_and_predict(frame)
        
        # Tampilkan frame
        FRAME_WINDOW.image(processed_frame, channels="BGR")
    
    cap.release()
    if not run_webcam:
        st.info("Webcam dihentikan")

# Catatan Kaki
st.markdown("---")
st.caption("Aplikasi deteksi emosi dengan Haar Cascade | Model: ONNX | Dibuat dengan Streamlit")
