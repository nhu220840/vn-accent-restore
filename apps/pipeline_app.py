import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import os
import threading
import sys
from PIL import Image, ImageDraw, ImageFont

# --- CẤU HÌNH PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Thêm project root vào sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import module thêm dấu
try:
    from src.utils.vn_accent_restore import restore_diacritics
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run from the project root.")
    exit()

# --- CẤU HÌNH ĐƯỜNG DẪN MODEL ---
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'model_mlp.pkl')

# --- BIẾN TOÀN CỤC ---
final_sentence = ""
is_processing_accent = False
cached_font_path = None  # Biến để lưu đường dẫn font sau khi tìm thấy

# --- [MỚI] HÀM TỰ ĐỘNG TÌM FONT TIẾNG VIỆT ---
def find_best_font_path():
    """
    Hàm này tự động quét hệ thống để tìm font hỗ trợ tiếng Việt.
    """
    global cached_font_path
    if cached_font_path and os.path.exists(cached_font_path):
        return cached_font_path

    # 1. Ưu tiên tìm file font.ttf ngay trong thư mục dự án
    local_paths = [
        "font.ttf",
        os.path.join(SCRIPT_DIR, "font.ttf"),
        os.path.join(PROJECT_ROOT, "font.ttf"),
    ]
    for path in local_paths:
        if os.path.exists(path):
            print(f"[Font] Found local font: {path}")
            cached_font_path = path
            return path

    # 2. Quét các thư mục font hệ thống của Ubuntu
    print("[Font] Scanning system for Vietnamese fonts...")
    
    # Danh sách font ưu tiên (hỗ trợ tiếng Việt tốt)
    target_fonts = [
        'Roboto-Regular.ttf', 
        'DejaVuSans.ttf', 
        'Ubuntu-R.ttf',
        'LiberationSans-Regular.ttf',
        'Arial.ttf',
        'FreeSans.ttf'
    ]
    
    # Các thư mục font chuẩn trên Linux
    system_font_dirs = [
        '/usr/share/fonts', 
        '/usr/local/share/fonts', 
        os.path.expanduser('~/.fonts')
    ]
    
    for font_dir in system_font_dirs:
        if not os.path.exists(font_dir): continue
        # Dùng os.walk để tìm sâu trong thư mục con
        for root, _, files in os.walk(font_dir):
            for filename in files:
                if filename in target_fonts:
                    found_path = os.path.join(root, filename)
                    print(f"[Font] Found system font: {found_path}")
                    cached_font_path = found_path
                    return found_path

    print("[Font] Warning: No specific Vietnamese font found. Using default.")
    return None

def get_font(size=30):
    font_path = find_best_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# --- HÀM VẼ CHỮ (SỬ DỤNG FONT TÌM ĐƯỢC) ---
def draw_vn_text(img, text, pos, font_size=30, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = get_font(font_size)
    
    # Thêm viền đen (stroke) để chữ dễ đọc hơn
    stroke_color = (0, 0, 0)
    stroke_width = 2
    
    draw.text(pos, text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_accent_async(raw_text):
    global final_sentence, is_processing_accent
    is_processing_accent = True
    try:
        # Chuẩn hóa input
        raw_text_lower = raw_text.lower().strip()
        print(f"Adding accents for: {raw_text_lower}")
        
        restored = restore_diacritics(raw_text_lower)
        
        # Chuẩn hóa output
        final_sentence = restored.capitalize()
        
    except Exception as e:
        print(f"Error adding accents: {e}")
        final_sentence = "Error!"
    finally:
        is_processing_accent = False

# --- LOAD MODEL ---
print("Loading Gesture Model and Scaler...")
if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model files not found at {MODEL_PATH}")
    exit()

try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- SETUP CAMERA ---
cap = cv2.VideoCapture(0)

current_sequence_raw = [] 
last_stable_prediction = None
stable_start_time = None
HOLD_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.7

was_hand_present = False 

# --- MAIN LOOP ---
WINDOW_NAME = 'Gesture Pipeline'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Tìm font 1 lần khi khởi động để cache
print("Initializing font system...")
find_best_font_path()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    detected_label = None
    is_hand_present = False 

    # 1. XỬ LÝ KHI CÓ TAY
    if results.multi_hand_landmarks:
        is_hand_present = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_landmarks.landmark[0]
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            if len(landmarks_list) == 63:
                input_data = np.array(landmarks_list).reshape(1, -1)
                scaled_data = scaler.transform(input_data)
                prediction = model.predict(scaled_data)[0]
                probabilities = model.predict_proba(scaled_data)[0]
                
                if np.max(probabilities) >= CONFIDENCE_THRESHOLD:
                    detected_label = str(prediction).lower()
                    
                    # Hiển thị nhãn đang nhận diện
                    cv2.putText(image_bgr, f"{detected_label.upper()}: {np.max(probabilities):.2f}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 2. PHÁT HIỆN BỎ TAY -> THÊM SPACE
    if was_hand_present and not is_hand_present:
        if current_sequence_raw and current_sequence_raw[-1] != " ":
            current_sequence_raw.append(" ")
            print("Hand removed -> Added SPACE")
            cv2.putText(image_bgr, "SPACE ADDED", (frame_w // 2 - 150, frame_h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    was_hand_present = is_hand_present

    # 3. LOGIC GIỮ TAY
    if detected_label:
        if detected_label == last_stable_prediction:
            elapsed = time.time() - stable_start_time
            
            bar_w = int((elapsed / HOLD_DURATION) * 200)
            cv2.rectangle(image_bgr, (10, 60), (10 + bar_w, 70), (0, 255, 0), -1)

            if elapsed >= HOLD_DURATION:
                if detected_label != " ": 
                    current_sequence_raw.append(detected_label)
                
                last_stable_prediction = None 
                stable_start_time = None
        else:
            last_stable_prediction = detected_label
            stable_start_time = time.time()
    else:
        last_stable_prediction = None

    # 4. HIỂN THỊ UI (GIỮ NGUYÊN TIẾNG ANH, DÙNG HÀM VẼ FONT XỊN)
    ui_height = 120
    cv2.rectangle(image_bgr, (0, frame_h - ui_height), (frame_w, frame_h), (30, 30, 30), -1)

    raw_text_combined = "".join(current_sequence_raw).strip()
    if raw_text_combined:
        raw_text_display = raw_text_combined[0].upper() + raw_text_combined[1:]
    else:
        raw_text_display = ""
        
    # Dùng draw_vn_text cho dòng Raw (đề phòng model trả về từ có dấu)
    image_bgr = draw_vn_text(image_bgr, f"Raw: {raw_text_display}", (20, frame_h - 80), font_size=30, color=(200, 200, 200))

    # Dòng Final: Đây là nơi quan trọng nhất cần hiển thị tiếng Việt
    final_display = f"Final: {final_sentence}"
    if is_processing_accent:
        final_display = "Final: Processing..."
    
    image_bgr = draw_vn_text(image_bgr, final_display, (20, frame_h - 40), font_size=35, color=(0, 255, 0))

    # Hướng dẫn (Tiếng Anh)
    guide_text = "'f': Fix Accents | 'c': Clear | 'q': Quit"
    cv2.putText(image_bgr, guide_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, image_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        current_sequence_raw = []
        final_sentence = ""
    elif key == ord('f'):
        if current_sequence_raw and not is_processing_accent:
            input_sentence = "".join(current_sequence_raw)
            threading.Thread(target=process_accent_async, args=(input_sentence,)).start()

cap.release()
cv2.destroyAllWindows()
hands.close()