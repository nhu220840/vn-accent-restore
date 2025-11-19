# =============================================
#  🧠 Huấn luyện MLP cho nhận dạng ngôn ngữ ký hiệu tiếng Việt
#  Input: train.csv, valid.csv, test.csv (cùng thư mục)
#  Output: model_mlp.pkl + scaler.pkl
# =============================================

import pandas as pd
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Đọc dữ liệu (CÁCH CHẠY ỔN ĐỊNH)
# -----------------------------

# --- Xác định vị trí tuyệt đối của script này ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Tạo đường dẫn tuyệt đối đến các file data ---
train_path = os.path.join(script_dir, r'..\1_data\processed\train_landmarks_augmented.csv')
valid_path = os.path.join(script_dir, r'..\1_data\processed\valid_landmarks.csv')
test_path = os.path.join(script_dir, r'..\1_data\processed\test_landmarks.csv')

# --- Đọc dữ liệu bằng đường dẫn mới ---
df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test = pd.read_csv(test_path)

print(f"Train: {len(df_train)} mẫu, Valid: {len(df_valid)}, Test: {len(df_test)}")

# -----------------------------
# 2️⃣ Chuẩn bị dữ liệu
# -----------------------------
X_train = df_train.drop('label', axis=1)
y_train = df_train['label']

X_valid = df_valid.drop('label', axis=1)
y_valid = df_valid['label']

X_test = df_test.drop('label', axis=1)
y_test = df_test['label']

# -----------------------------
# 3️⃣ Chuẩn hóa dữ liệu
# -----------------------------
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print("✅ Dữ liệu đã được chuẩn hóa (StandardScaler).")

# -----------------------------
# 4️⃣ Huấn luyện MLP (với GridSearch nhẹ)
# -----------------------------
param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128, 64)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-4, 1e-3],  # regularization
    'learning_rate_init': [0.001, 0.0005]
}

print("🔍 Đang tìm cấu hình tốt nhất bằng GridSearchCV (mất vài phút)...")
mlp = MLPClassifier(max_iter=300, random_state=42)
grid = GridSearchCV(mlp, param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print(f"✅ Cấu hình tốt nhất: {grid.best_params_}")

# -----------------------------
# 5️⃣ Đánh giá trên tập VALID
# -----------------------------
valid_preds = best_model.predict(X_valid_scaled)
valid_acc = accuracy_score(y_valid, valid_preds)
print(f"\n🎯 Accuracy (VALID): {valid_acc * 100:.2f}%")

print("\n📊 Báo cáo chi tiết:")
print(classification_report(y_valid, valid_preds))

# -----------------------------
# 6️⃣ Đánh giá trên tập TEST
# -----------------------------
test_preds = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)
print(f"\n🧾 Accuracy (TEST): {test_acc * 100:.2f}%")

# -----------------------------
# 7️⃣ Ma trận nhầm lẫn
# -----------------------------
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# -----------------------------
# 8️⃣ Lưu model và scaler
# -----------------------------
# --- Tạo đường dẫn lưu file tuyệt đối ---
MODEL_PATH = os.path.join(script_dir, r'..\4_models\model_mlp.pkl')
SCALER_PATH = os.path.join(script_dir, r'..\4_models\scaler.pkl')

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n💾 Đã lưu model: {MODEL_PATH}")
print(f"💾 Đã lưu scaler: {SCALER_PATH}")
print("🚀 Huấn luyện hoàn tất!")