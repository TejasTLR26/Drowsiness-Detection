import cv2
import numpy as np
import os
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from pygame import mixer

# Initialize alert sound # final file
mixer.init()
sound = mixer.Sound('alarm2.wav')

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load trained eye state model
model = load_model(os.path.join(r'C:\Users\aayue\OneDrive\Desktop\Projects\Drowsiness Detection\models\model.h5'))

# Labels
lbl = ['Close', 'Open']

# Mediapipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Lip landmark indices (outer)
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Yawn detection landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
YAWN_THRESHOLD = 30

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color_open = (0, 255, 0)
font_color_closed = (0, 0, 255)
font_thickness = 2

# Drowsiness tracking
score = 0
alert_display = False
alert_start_time = None
alert_duration = 2  # in seconds

# Lip tracking
prev_lip_coords = None
lip_movement_threshold = 1.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Face and Eye detection using Haar
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    for (x, y, w_face, h_face) in faces:
        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)

    for (x, y, w_eye, h_eye) in eyes:
        eye = frame[y:y + h_eye, x:x + w_eye]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)

        if prediction[0][0] > 0.30:  # Eye Closed
            score += 1
            cv2.putText(frame, "Closed", (20, 50), font, font_scale, font_color_closed, font_thickness, cv2.LINE_AA)

            if score > 7 and not alert_display:
                sound.play()
                alert_display = True
                alert_start_time = time.time()

        elif prediction[0][1] > 0.70:  # Eye Open
            score = max(0, score - 1)
            cv2.putText(frame, "Open", (20, 50), font, font_scale, font_color_open, font_thickness, cv2.LINE_AA)

    # Drowsiness Alert Message
    if alert_display:
        elapsed_time = time.time() - alert_start_time
        if elapsed_time < alert_duration:
            cv2.putText(frame, "ALERT! EYES CLOSED!", (150, 100), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            alert_display = False

    # Mediapipe face mesh processing
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Yawn Detection
            upper_lip = landmarks[UPPER_LIP]
            lower_lip = landmarks[LOWER_LIP]
            yawn_dist = abs((lower_lip.y * h) - (upper_lip.y * h))
            if yawn_dist > YAWN_THRESHOLD:
                cv2.putText(frame, "YAWNING! PLEASE REST", (130, 200), font, 0.9, (0, 140, 255), 3)

            # Lip Movement Detection
            lip_coords = []
            for idx in LIP_LANDMARKS:
                pt = landmarks[idx]
                lip_coords.append((pt.x * w, pt.y * h))

            lip_coords = np.array(lip_coords)
            if prev_lip_coords is not None:
                movement = np.linalg.norm(lip_coords - prev_lip_coords, axis=1)
                avg_movement = np.mean(movement)

                if avg_movement > lip_movement_threshold:
                    cv2.putText(frame, "Lip Movement Detected", (150, 250), font, 0.8, (255, 0, 255), 2)
            prev_lip_coords = lip_coords

    # Display instructions
    cv2.putText(frame, "Press 'E' to exit", (450, 470), font, 0.6, (255, 255, 255), 1)

    # Show frame
    cv2.imshow('Drowsiness + Yawn + Lip Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
