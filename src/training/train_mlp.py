import pandas as pd
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

train_path = os.path.join(project_root, 'data', 'processed', 'train_landmarks_augmented.csv')
valid_path = os.path.join(project_root, 'data', 'processed', 'valid_landmarks.csv')
test_path = os.path.join(project_root, 'data', 'processed', 'test_landmarks.csv')

df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test = pd.read_csv(test_path)

print(f"Train: {len(df_train)} samples, Valid: {len(df_valid)}, Test: {len(df_test)}")

X_train = df_train.drop('label', axis=1)
y_train = df_train['label']

X_valid = df_valid.drop('label', axis=1)
y_valid = df_valid['label']

X_test = df_test.drop('label', axis=1)
y_test = df_test['label']
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print("Data normalized (StandardScaler).")

param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128, 64)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-4, 1e-3],
    'learning_rate_init': [0.001, 0.0005]
}

print("Searching for best configuration using GridSearchCV (takes a few minutes)...")
mlp = MLPClassifier(max_iter=300, random_state=42)
grid = GridSearchCV(mlp, param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print(f"Best configuration: {grid.best_params_}")

valid_preds = best_model.predict(X_valid_scaled)
valid_acc = accuracy_score(y_valid, valid_preds)
print(f"\nAccuracy (VALID): {valid_acc * 100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_valid, valid_preds))

test_preds = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)
print(f"\nAccuracy (TEST): {test_acc * 100:.2f}%")

MODEL_PATH = os.path.join(project_root, 'models', 'model_mlp.pkl')
SCALER_PATH = os.path.join(project_root, 'models', 'scaler.pkl')

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nModel saved: {MODEL_PATH}")
print(f"Scaler saved: {SCALER_PATH}")
print("Training completed!")