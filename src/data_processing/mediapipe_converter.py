# --- PART 1: SETUP (Including Albumentations) ---
import mediapipe as mp
import cv2
import os
import json
import pandas as pd
import numpy as np
import albumentations as A  # Augmentation library

# 1.1. Import only solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 1.2. Define Augmentation "recipe"
# (This recipe will ONLY apply to the Train set)
augment_pipeline = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# 1.3. Number of "virtual copies" to create PER image
# === CHANGE 1: INCREASE AUGMENTATION COUNT ===
# You can change this number 9 to any number you want
N_AUGMENTATIONS_PER_IMAGE = 9 
# === END OF CHANGE 1 ===

print(f"Imported libraries and defined Augmentation Pipeline (N={N_AUGMENTATIONS_PER_IMAGE} copies per image).")

# --- PART 2: DEFINE PROCESSING FUNCTION ---
# Add 'is_training' flag to know when to augment
def process_dataset(json_path, image_dir, is_training=False):
    
    print(f"\n[Start] Processing dataset at: {json_path}")
    # Print Augmentation status
    print(f"  -> Augmentation: {'ON' if is_training else 'OFF'} ({N_AUGMENTATIONS_PER_IMAGE} copies)" if is_training else f"  -> Augmentation: {'OFF'}")

    # Create a NEW instance for each set (train/valid/test)
    hands_processor = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    print("  -> Initialized NEW MediaPipe instance.")
    
    # Open and read COCO JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create lookup map (unchanged)
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_label_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_filename_to_label = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        filename = image_id_to_filename[image_id]
        label = category_id_to_label_name[category_id]
        image_filename_to_label[filename] = label

    processed_data_list = []
    total_images = len(image_filename_to_label)
    print(f"  -> Found {total_images} original images to process.")

    for i, (filename, label) in enumerate(image_filename_to_label.items()):
        
        if (i + 1) % 500 == 0: 
            print(f"    ...Processed {i+1} / {total_images} original images...")

        # Read image (Using np.fromfile)
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        try:
            n = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception:
            continue
        if image is None:
             continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create list of images to process
        images_to_process = []
        images_to_process.append(image_rgb) # 1. Always add the ORIGINAL image
        
        # 2. ONLY if training set, create "virtual" images
        if is_training:
            for _ in range(N_AUGMENTATIONS_PER_IMAGE):
                augmented = augment_pipeline(image=image_rgb)
                images_to_process.append(augmented['image'])
        
        # Loop through (1 original image) or (N+1 images)
        for img_to_process in images_to_process:
            results = hands_processor.process(img_to_process)
            if results.multi_hand_landmarks:
                # (Code for extracting 63 points remains unchanged)
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist_landmark = hand_landmarks.landmark[0]
                relative_landmarks_flat = []
                for landmark in hand_landmarks.landmark:
                    relative_landmarks_flat.append(landmark.x - wrist_landmark.x)
                    relative_landmarks_flat.append(landmark.y - wrist_landmark.y)
                    relative_landmarks_flat.append(landmark.z - wrist_landmark.z)
                row = [label] + relative_landmarks_flat
                processed_data_list.append(row)
            else:
                pass # Skip silently

    print(f"[Completed] Extracted {len(processed_data_list)} valid samples.")
    hands_processor.close()
    return processed_data_list

# --- PART 3: DEFINE PATHS & RUN ---
print(f"\nCurrent working directory: {os.getcwd()}")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
base_data_directory = os.path.join(project_root, "data", "raw", "data_mini_app_01")
print(f"Base data directory: {base_data_directory}") 

# Train paths
train_json_path = os.path.join(base_data_directory, "train", "_annotations.coco.json")
train_image_directory = os.path.join(base_data_directory, "train")
# Valid paths
valid_json_path = os.path.join(base_data_directory, "valid", "_annotations.coco.json")
valid_image_directory = os.path.join(base_data_directory, "valid")
# EDIT: Add Test paths
test_json_path = os.path.join(base_data_directory, "test", "_annotations.coco.json")
test_image_directory = os.path.join(base_data_directory, "test")

# Check paths
if not os.path.exists(train_json_path):
    print(f"\n--- ERROR! JSON FILE NOT FOUND (Train) ---")
    exit()
else:
    print(f"OK! Found train JSON file.")

# Run processing
# is_training=True -> Augmentation ON
train_data_rows = process_dataset(train_json_path, train_image_directory, is_training=True)
# is_training=False -> Augmentation OFF
valid_data_rows = process_dataset(valid_json_path, valid_image_directory, is_training=False)
# EDIT: Run processing for Test set (Augmentation OFF)
test_data_rows = process_dataset(test_json_path, test_image_directory, is_training=False)


# --- PART 4: FILTER AND SAVE RESULTS TO CSV ---
columns = ['label'] 
for i in range(21):
    columns += [f'x{i}', f'y{i}', f'z{i}']
print(f"\nPreparing to save CSV file with {len(columns)} columns.")

df_train = pd.DataFrame(train_data_rows, columns=columns)
df_valid = pd.DataFrame(valid_data_rows, columns=columns)
df_test = pd.DataFrame(test_data_rows, columns=columns)

# === CHANGE 2: FILTER OUT 'DD' LABEL BEFORE SAVING ===
print("\n[Start] Filtering 'DD' label...")

len_before_train = len(df_train)
df_train = df_train[df_train['label'] != 'DD'].reset_index(drop=True)
len_after_train = len(df_train)
print(f"  -> Train: Filtered {len_before_train - len_after_train} 'DD' samples. Remaining: {len_after_train} samples.")

len_before_valid = len(df_valid)
df_valid = df_valid[df_valid['label'] != 'DD'].reset_index(drop=True)
len_after_valid = len(df_valid)
print(f"  -> Valid: Filtered {len_before_valid - len_after_valid} 'DD' samples. Remaining: {len_after_valid} samples.")

len_before_test = len(df_test)
df_test = df_test[df_test['label'] != 'DD'].reset_index(drop=True)
len_after_test = len(df_test)
print(f"  -> Test:  Filtered {len_before_test - len_after_test} 'DD' samples. Remaining: {len_after_test} samples.")

print("[Completed] Finished filtering 'DD' label.")
# === END OF CHANGE 2 ===


output_dir = os.path.join(project_root, "data", "processed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\nCreated directory: {output_dir}")
else:
    print(f"\nOutput directory: {output_dir}")

train_csv_filename = os.path.join(output_dir, "train_landmarks_augmented.csv") 
valid_csv_filename = os.path.join(output_dir, "valid_landmarks.csv")
test_csv_filename = os.path.join(output_dir, "test_landmarks.csv")

df_train.to_csv(train_csv_filename, index=False)
df_valid.to_csv(valid_csv_filename, index=False)
df_test.to_csv(test_csv_filename, index=False) # EDIT: Save test file

# --- PART 5: CLEANUP AND FINISH (EDIT: Add TEST) ---
print("\n--- ALL DONE! ---")
print(f"Saved train data to file: {train_csv_filename} ({len(df_train)} rows)")
print(f"Saved valid data to file: {valid_csv_filename} ({len(df_valid)} rows)")
print(f"Saved test data to file: {test_csv_filename} ({len(df_test)} rows)")