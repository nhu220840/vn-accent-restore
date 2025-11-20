import cv2
import joblib
import mediapipe as mp
import numpy as np
import time

import os

print("Loading model and scaler... (please wait)")
try:
    # Cross-platform path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    model_path = os.path.join(project_root, 'models', 'model_mlp.pkl')
    
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    MODEL_CLASSES = model.classes_
    N_CLASSES = len(MODEL_CLASSES)
    
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print("Please check the file path and name.")
    exit()
    
print("Loaded successfully! Starting webcam.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_sequence = []
sequence_history = []

last_stable_prediction = None
stable_start_time = None
HOLD_DURATION = 1.75 

NO_HAND_DURATION = 2.75
no_hand_start_time = None

CONFIDENCE_THRESHOLD = 0.7

WINDOW_NAME = 'Webcam: \'q\'=quit, \'s\'=save word, \'c\'=clear word'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    label = None 

    bar_area_width = 300  
    bar_area = np.zeros((frame_h, bar_area_width, 3), dtype="uint8")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            wrist_z = hand_landmarks.landmark[0].z
            
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.append(lm.x - wrist_x)
                landmarks_list.append(lm.y - wrist_y)
                landmarks_list.append(lm.z - wrist_z)

            if len(landmarks_list) == 63:
                  
                input_data = np.array(landmarks_list).reshape(1, -1)
                scaled_data = scaler.transform(input_data)
                
                prediction = model.predict(scaled_data)
                probabilities = model.predict_proba(scaled_data)[0]

                max_prob = np.max(probabilities) 
                
                if max_prob >= CONFIDENCE_THRESHOLD:
                    label = prediction[0]  
                else:
                    label = None  
                
                bar_height_per_class = frame_h // N_CLASSES
                
                for i in range(N_CLASSES):
                    class_name = MODEL_CLASSES[i]
                    prob = probabilities[i]  
                    
                    bar_width = int(prob * (bar_area_width - 10))
                    y_start = i * bar_height_per_class
                    y_end = (i + 1) * bar_height_per_class

                    color = (0, 100, 0)
                    text_color = (255, 255, 255)
                    
                    if class_name == label: 
                        color = (0, 255, 255)
                        text_color = (0, 0, 0)
                    
                    cv2.rectangle(bar_area, (5, y_start + 2), (5 + bar_width, y_end - 2), color, -1)
                    text = f"{class_name}: {prob*100:.0f}%"
                    cv2.putText(bar_area, text, (10, y_end - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                cv2.putText(image_bgr, f"PRED: {label}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
            
    timer_text = ""
    
    if label is not None:
        no_hand_start_time = None 
        
        if label == last_stable_prediction:
            elapsed = time.time() - stable_start_time
            timer_text = f"Hold {label} ({elapsed:.1f}s)"
            
            progress_w = int((elapsed / HOLD_DURATION) * (frame_w - 20))
            if progress_w > (frame_w - 20): progress_w = frame_w - 20
            cv2.rectangle(image_bgr, (10, 110), (10 + progress_w, 120), (0, 255, 255), -1)
            
            if elapsed > HOLD_DURATION:
                current_sequence.append(label)
                last_stable_prediction = None 
                stable_start_time = None
                timer_text = f"ADDED: {label}"
        else:
            last_stable_prediction = label
            stable_start_time = time.time()
            timer_text = f"Detected {label} (0.0s)"
            
    else:
        last_stable_prediction = None
        stable_start_time = None
        
        if no_hand_start_time is None:
            no_hand_start_time = time.time()
            timer_text = "" 
        else:
            no_hand_elapsed = time.time() - no_hand_start_time
            
            progress_w = int((no_hand_elapsed / NO_HAND_DURATION) * (frame_w - 20))
            if progress_w > (frame_w - 20): progress_w = frame_w - 20
            cv2.rectangle(image_bgr, (10, 110), (10 + progress_w, 120), (0, 255, 0), -1)
            
            if no_hand_elapsed > NO_HAND_DURATION:
                if current_sequence and current_sequence[-1] != " ":
                    current_sequence.append(" ")
                    timer_text = "ADDED: Space"
                
                no_hand_start_time = None 
                
            else:
                timer_text = f"Waiting for space ({no_hand_elapsed:.1f}s)"

    cv2.putText(image_bgr, timer_text, (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
    sequence_text = "".join(current_sequence)
    cv2.rectangle(image_bgr, (0, frame_h - 60), (frame_w, frame_h), (0,0,0), -1)
    cv2.putText(image_bgr, sequence_text, (20, frame_h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    history_y_start = (N_CLASSES * (frame_h // N_CLASSES)) + 30
    if history_y_start > frame_h - 100: 
        history_y_start = 20
        
    cv2.putText(bar_area, "HISTORY ('s'=save):", (10, history_y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i, word in enumerate(reversed(sequence_history[-5:])):
        history_text = f"- {word}"
        cv2.putText(bar_area, history_text, (10, history_y_start + 25 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    final_frame = np.concatenate((image_bgr, bar_area), axis=1)

    cv2.imshow(WINDOW_NAME, final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
    if key == ord('s'):
        if current_sequence:  
            sequence_history.append("".join(current_sequence))
            current_sequence = []  
            
    if key == ord('c'):
        current_sequence = []
    
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()