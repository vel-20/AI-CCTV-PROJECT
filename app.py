```python
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import time

from mtcnn import MTCNN
from keras_facenet import FaceNet

st.set_page_config(page_title="AI CCTV PRO", layout="wide")

st.title("🔍 AI CCTV Surveillance System")

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource
def load_models():
    try:
        fight_model = tf.keras.models.load_model("mobilenet_fight.h5", compile=False)
    except:
        fight_model = None

    yolo_model = YOLO("yolov8n.pt")
    detector = MTCNN()
    embedder = FaceNet()

    return fight_model, yolo_model, detector, embedder

fight_model, yolo_model, detector, embedder = load_models()

IMG_SIZE = 224

# ------------------------------
# SIDEBAR
# ------------------------------
start = st.sidebar.checkbox("Start Camera")

uploaded_file = st.sidebar.file_uploader("Upload Reference Face", type=["jpg", "png"])

reference_embedding = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    reference_img = cv2.imdecode(file_bytes, 1)

    st.sidebar.image(reference_img, caption="Reference Face")

    rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)

        face = rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160,160))
        reference_embedding = embedder.embeddings([face])[0]

frame_window = st.image([])

prediction_history = deque(maxlen=20)
last_alert_time = 0

# ------------------------------
# CAMERA LOOP
# ------------------------------
if start:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        frame = cv2.resize(frame, (640,480))

        # ------------------------------
        # YOLO PERSON DETECTION
        # ------------------------------
        results = yolo_model(frame, verbose=False)
        person_count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    person_count += 1
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # ------------------------------
        # MOTION DETECTION
        # ------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        if 'prev' not in st.session_state:
            st.session_state.prev = gray

        diff = cv2.absdiff(st.session_state.prev, gray)
        motion_score = np.sum(diff) / 255
        motion_detected = motion_score > 5000

        st.session_state.prev = gray

        # ------------------------------
        # FIGHT DETECTION
        # ------------------------------
        abnormal = False
        confidence = 0

        if fight_model is not None:
            inp = cv2.resize(frame,(IMG_SIZE,IMG_SIZE)) / 255.0
            inp = np.expand_dims(inp,0)

            confidence = float(fight_model.predict(inp,verbose=0)[0][0])
            prediction_history.append(confidence)

            threshold = np.mean(prediction_history) + 0.1 if len(prediction_history)>5 else 0.6
            abnormal = confidence > threshold and motion_detected

        # ------------------------------
        # ALERT
        # ------------------------------
        if abnormal:
            last_alert_time = time.time()

        alert = (time.time() - last_alert_time) < 3

        if alert:
            cv2.putText(frame,"ALERT!",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        # ------------------------------
        # FACE RECOGNITION
        # ------------------------------
        if reference_embedding is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            for f in faces:
                x,y,w,h = f['box']
                x,y = max(0,x), max(0,y)

                face = rgb[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                face = cv2.resize(face,(160,160))
                emb = embedder.embeddings([face])[0]

                dist = np.linalg.norm(reference_embedding - emb)

                label = "MATCH" if dist < 0.9 else "Unknown"
                color = (0,255,0) if dist < 0.9 else (255,255,0)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # ------------------------------
        # DISPLAY
        # ------------------------------
        frame_window.image(frame, channels="BGR")

    cap.release()
```
