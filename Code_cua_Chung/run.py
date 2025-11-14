import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ==============================
# 🔹 1. Load model và scaler
# ==============================
MODEL_PATH = 'model_mlp.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    print("🔄 Đang tải model và scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Đã tải model và scaler!")
except Exception as e:
    print(f"❌ Không thể tải model hoặc scaler: {e}")
    exit()

# ==============================
# 🔹 2. Khởi tạo Mediapipe
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ==============================
# 🔹 3. Biến điều khiển
# ==============================
sentence_raw = ""
last_detection_time = time.time()
last_recognition_time = 0
running = True
cap = None
is_scanning = True # === SỬA ĐỘ TRỄ 1: Thêm biến trạng thái ===

# === [WIDGETS TOÀN CỤC] ===
label_text_model2 = None
frame_separator = None
frame_bottom_right = None
frame_buttons_scanning = None
frame_buttons_review = None
# =======================================

# ==============================
# 🔹 4. Hàm xử lý
# ==============================
def reset_text_scanning():
    global sentence_raw
    sentence_raw = ""
    label_text.set("")
    print("\n--- KẾT QUẢ QUÉT ĐÃ ĐƯỢC RESET ---")

def quit_app():
    global running, cap
    running = False
    # cho camera có thời gian thoát vòng lặp
    time.sleep(0.2)
    try:
        if cap:
            cap.release()
    except Exception:
        pass
    try:
        if root and root.winfo_exists():
            root.destroy()
    except Exception:
        pass
    print("Ứng dụng đã đóng (quit_app).")

def recognize_again():
    """
    Quay về trạng thái trước khi bấm 'Thêm dấu':
    - Ẩn phần model2 + separator
    - Ẩn bộ nút review, hiện lại bộ nút scanning
    - Reset text hiển thị của cả 2
    - Bật lại chế độ quét (KHÔNG khởi động lại camera)
    """
    global running, sentence_raw, is_scanning # === SỬA ĐỘ TRỄ 2 ===

    # Ẩn phần model 2 và separator
    frame_separator.pack_forget()
    frame_bottom_right.pack_forget()

    # Xóa text
    sentence_raw = ""
    label_text.set("")
    label_text_model2.set("")

    # Ẩn bộ nút review, hiện bộ nút quét
    try:
        frame_buttons_review.pack_forget()
    except Exception:
        pass
    frame_buttons_scanning.pack(fill='x')

    # === SỬA ĐỘ TRỄ 2: Chỉ cần bật lại biến is_scanning ===
    # (Xóa toàn bộ khối 'if not running' và 'threading.Thread')
    is_scanning = True
    print("🔄 Quay lại trạng thái quét (camera vẫn chạy).")
    # === KẾT THÚC SỬA 2 ===

def process_model_2():
    """
    Khi bấm 'Thêm dấu':
    - Dừng quét (đặt is_scanning = False) - KHÔNG DỪNG CAMERA
    - Hiển thị phần 'Sau khi thêm dấu' với text
    - Hiện bộ nút review (Nhận diện lại / Thoát)
    - Ẩn bộ nút quét gốc
    """
    global running, sentence_raw, label_text_model2, is_scanning # === SỬA ĐỘ TRỄ 3 ===

    # === SỬA ĐỘ TRỄ 3: Dừng quét, không dừng thread ===
    # (Thay 'running = False' bằng 'is_scanning = False')
    is_scanning = False 
    # (Xóa time.sleep(0.15))
    # === KẾT THÚC SỬA 3 ===

    # Lấy kết quả hiện tại (không thay đổi chữ hoa/chữ thường)
    final_text = sentence_raw
    label_text_model2.set(final_text)
    print(f"✅ Kết quả cuối cùng: {final_text}")

    # Hiện phần separator + model 2 (chia đôi khung phải)
    frame_separator.pack(fill='x', pady=(10, 5))
    frame_bottom_right.pack(fill='x', expand=True)

    # Ẩn bộ nút quét
    try:
        frame_buttons_scanning.pack_forget()
    except Exception:
        pass

    # Hiện bộ nút review (2 nút bạn muốn)
    frame_buttons_review.pack(fill='x', pady=10)

    # đảm bảo UI vẽ lại ngay
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pass

# ==============================
# 🔹 5. Hàm xử lý Camera
# ==============================
def camera_loop():
    # === SỬA ĐỘ TRỄ 4: Thêm 'is_scanning' vào global ===
    global last_detection_time, last_recognition_time, sentence_raw, cap, running, is_scanning

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera.")
        return
    print("Camera đã khởi động.")

    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    except Exception:
        pass

    while running:
        ret, frame = cap.read()
        if not ret or not running:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # === SỬA ĐỘ TRỄ 4: Toàn bộ khối xử lý ảnh giờ nằm trong 'if is_scanning:' ===
        # Frame gốc (chưa vẽ) để hiển thị khi không quét
        final_frame_for_gui = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if is_scanning:
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            current_time = time.time()
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                drawing_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    drawing_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                all_landmarks_list = hand_landmarks.landmark
                base_x, base_y, base_z = all_landmarks_list[0].x, all_landmarks_list[0].y, all_landmarks_list[0].z

                landmarks_relative = []
                for lm in all_landmarks_list:
                    landmarks_relative.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                # Giới hạn tần suất dự đoán (2s)
                if current_time - last_recognition_time >= 2.0:
                    try:
                        X_input = np.array(landmarks_relative).reshape(1, -1)
                        X_scaled = scaler.transform(X_input)
                        y_pred = model.predict(X_scaled)
                        detected_letter = y_pred[0]
                        last_recognition_time = current_time

                        sentence_raw += detected_letter
                        label_text.set(sentence_raw)
                    except Exception as e:
                        print(f"Lỗi khi dự đoán: {e}")

                last_detection_time = time.time()
                final_frame_for_gui = drawing_frame # Cập nhật frame để vẽ
            else:
                # Nếu không phát hiện tay trong >2.5s => thêm dấu cách
                if current_time - last_detection_time > 2.5:
                    if len(sentence_raw) > 0 and not sentence_raw.endswith(" "):
                        sentence_raw += " "
                        label_text.set(sentence_raw)
                    last_detection_time = current_time
        # === KẾT THÚC SỬA 4 (kết thúc khối 'if is_scanning:') ===

        try:
            # Khối hiển thị camera này luôn chạy
            display_frame = cv2.flip(final_frame_for_gui, 1)
            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)

            if running:
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
        except Exception as e:
            if running:
                print(f"Lỗi cập nhật GUI: {e}")

    # Thoát camera khi vòng lặp dừng (khi bấm 'Thoát')
    try:
        if cap:
            cap.release()
    except Exception:
        pass
    print("Camera loop đã dừng.")

# ==============================
# 🔹 6. Giao diện Tkinter
# ==============================
root = tk.Tk()
root.title("Vietnamese Sign Language Recognition")
root.geometry("1280x480")
root.resizable(False, False)

# --- Khung trái (camera) ---
frame_left = tk.Frame(root, width=640, height=480, bg="black")
frame_left.pack(side="left", fill="both", expand=True)
frame_left.pack_propagate(False)
video_label = tk.Label(frame_left, bg="black")
video_label.pack(fill="both", expand=True)

# --- Khung phải (text + nút) ---
frame_right = tk.Frame(root, width=640, height=480, bg="#1E1E1E")
frame_right.pack(side="right", fill="both", expand=True)
frame_right.pack_propagate(False)

# (Đã sửa lỗi layout từ lần trước)
frame_content = tk.Frame(frame_right, bg="#1E1E1E")
btn_frame = tk.Frame(frame_right, bg="#1E1E1E")
btn_frame.pack(side='bottom', fill='x', pady=20)
frame_content.pack(fill='both', expand=True, side='top')

# --- Phần trên (Model 1) ---
label_title = tk.Label(frame_content, text="Kết quả nhận diện", font=("Arial", 18, "bold"), fg="white", bg="#1E1E1E")
label_title.pack(pady=(20, 10))
text_display_frame = tk.Frame(frame_content, bg="#1E1E1E", height=150, width=600)
text_display_frame.pack(padx=20)
text_display_frame.pack_propagate(False)
label_text = tk.StringVar()
label_display = tk.Label(text_display_frame, textvariable=label_text, font=("Consolas", 20), fg="#00FF00", bg="#1E1E1E", wraplength=580, justify="left", anchor="nw")
label_display.pack(fill="both", expand=True, padx=10)

# --- Phần Ngăn cách (Ẩn ban đầu) ---
frame_separator = tk.Frame(frame_content, bg="#1E1E1E")
sep1 = tk.Frame(frame_separator, height=2, bg='gray50')
sep1.pack(fill='x', padx=20, pady=5)
sep2 = tk.Frame(frame_separator, height=2, bg='gray50')
sep2.pack(fill='x', padx=20)

# --- Phần dưới (Model 2 - Ẩn ban đầu) ---
frame_bottom_right = tk.Frame(frame_content, bg="#1E1E1E")
label_title_model2 = tk.Label(frame_bottom_right, text="Sau khi thêm dấu", font=("Arial", 18, "bold"), fg="white", bg="#1E1E1E")
label_title_model2.pack(pady=(10, 10))
text_display_frame_model2 = tk.Frame(frame_bottom_right, bg="#1E1E1E", height=150, width=600)
text_display_frame_model2.pack(padx=20)
text_display_frame_model2.pack_propagate(False)
label_text_model2 = tk.StringVar()
label_display_model2 = tk.Label(text_display_frame_model2, textvariable=label_text_model2, font=("Consolas", 20), fg="#00FF00", bg="#1E1E1E", wraplength=580, justify="left", anchor="nw")
label_display_model2.pack(fill="both", expand=True, padx=10)

# --- Khung Nút Bấm ---
# --- [Bộ nút 1: Đang quét] ---
frame_buttons_scanning = tk.Frame(btn_frame, bg="#1E1E1E")
frame_buttons_scanning.pack(fill='x')  # Hiển thị ban đầu
frame_buttons_scanning.columnconfigure(0, weight=1)
frame_buttons_scanning.columnconfigure(1, weight=1)
frame_buttons_scanning.columnconfigure(2, weight=1)

btn_reset = tk.Button(frame_buttons_scanning, text="🔁 Reset", command=reset_text_scanning, width=10, height=2, bg="#007ACC", fg="white", font=("Arial", 12, "bold"))
btn_reset.grid(row=0, column=0, sticky='e', padx=10)
btn_add_diacritics = tk.Button(frame_buttons_scanning, text="✅ Thêm dấu", command=process_model_2, width=10, height=2, bg="#5CB85C", fg="white", font=("Arial", 12, "bold"))
btn_add_diacritics.grid(row=0, column=1, sticky='', padx=10)
btn_quit = tk.Button(frame_buttons_scanning, text="❌ Thoát", command=quit_app, width=10, height=2, bg="#D9534F", fg="white", font=("Arial", 12, "bold"))
btn_quit.grid(row=0, column=2, sticky='w', padx=10)

# --- [Bộ nút 2: Đang xem lại] (định nghĩa, không pack ban đầu) ---
frame_buttons_review = tk.Frame(btn_frame, bg="#1E1E1E")
frame_buttons_review.columnconfigure(0, weight=1)
frame_buttons_review.columnconfigure(1, weight=1)

btn_recognize_again = tk.Button(frame_buttons_review, text="🔄 Nhận diện lại", command=recognize_again, width=15, height=2, bg="#007ACC", fg="white", font=("Arial", 12, "bold"))
btn_recognize_again.grid(row=0, column=0, sticky='e', padx=20)
btn_quit_2 = tk.Button(frame_buttons_review, text="❌ Thoát", command=quit_app, width=15, height=2, bg="#D9534F", fg="white", font=("Arial", 12, "bold"))
btn_quit_2.grid(row=0, column=1, sticky='w', padx=20)

# Kết thúc cấu hình UI
root.protocol("WM_DELETE_WINDOW", quit_app)

# Hỏi người dùng bật camera
if messagebox.askyesno("Bật camera", "📷 Bạn có cho phép mở camera để nhận diện tay không?"):
    # Chỉ khởi động thread này MỘT LẦN DUY NHẤT
    threading.Thread(target=camera_loop, daemon=True).start()
else:
    messagebox.showinfo("Thoát", "Bạn đã từ chối bật camera. Ứng dụng sẽ đóng.")
    root.destroy()

if 'root' in locals() and root.winfo_exists():
    root.mainloop()

print("Ứng dụng đã đóng.")