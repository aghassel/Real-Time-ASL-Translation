# %% [markdown]
# # Real-time Webcam to ASL Translation

# %% [markdown]
# Model - combination of a 1D CNN and a Transformer, and used 4x seed ensemble for submission
# From: https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
# 
# Latency: 17ms

# %%
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import json

ROWS_PER_FRAME = 543  # number of holistic landmarks
data_columns = 3  # 'x', 'y', 'z' for each landmark
BUFFER_SIZE = 5  # number of frames to buffer before making a prediction

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, 
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_label_map(json_file_path):
    with open(json_file_path, 'r') as file:
        label_map = json.load(file)
    index_to_label = {v: k for k, v in label_map.items()}
    return index_to_label

index_to_label = load_label_map("sign_to_prediction_index_map.json")

# %%
def extract_landmarks(results):
    landmarks = {'face': results.face_landmarks, 'left_hand': results.left_hand_landmarks,
                 'pose': results.pose_landmarks, 'right_hand': results.right_hand_landmarks}
    all_landmarks = []
    for key, result in landmarks.items():
        num_landmarks = {'face': 468, 'left_hand': 21, 'pose': 33, 'right_hand': 21}[key]
        if result is None:
            all_landmarks.extend([(0, 0, 0)] * num_landmarks)
        else:
            all_landmarks.extend([(landmark.x, landmark.y, landmark.z) for landmark in result.landmark])
    return all_landmarks

def update_buffer(landmarks_buffer, new_landmarks, buffer_size):
    landmarks_buffer.append(new_landmarks)
    if len(landmarks_buffer) > buffer_size:
        landmarks_buffer.pop(0)
    
    return landmarks_buffer

landmarks_buffer = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
    landmarks = extract_landmarks(results)
    landmarks_buffer = update_buffer(landmarks_buffer, landmarks, BUFFER_SIZE)
    predicted_label = None
    labels = None
    p = None
    predicted_index = None
    confidence = None
    if len(landmarks_buffer) == BUFFER_SIZE:
        flat_list = [item for sublist in landmarks_buffer for item in sublist]
        df = pd.DataFrame(flat_list, columns = ['x', 'y', 'z'])
        n_frames = int(len(df) / ROWS_PER_FRAME)
        df = df.values.reshape(n_frames, ROWS_PER_FRAME, len(df.columns))
        df = df.astype(np.float32)
        prediction_fcn = interpreter.get_signature_runner('serving_default')
        output = prediction_fcn(inputs=df)
        p = output['outputs'].reshape(-1)
        p_normalized = np.exp(p) / np.sum(np.exp(p))
        predicted_index = np.argmax(p)
        confidence = p_normalized[predicted_index]

        if confidence > 0.25:
            predicted_label = index_to_label[predicted_index]
        else:
            predicted_label = None
    if confidence:
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Word Translation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



