# Gesture2Text Model

A hand gesture recognition system that converts hand gestures to text using MediaPipe and Machine Learning.

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python apps/webcam_app.py
```

## 📋 Requirements

- **Python**: 3.8+ (recommended 3.10-3.12)
- **OS**: Windows, Linux, or macOS
- **Webcam**: For real-time application
- **RAM**: Minimum 4GB (recommended 8GB+)
- **Storage**: ~2GB for data and models

## 📁 Project Structure

```
Gesture2Text_Model/
├── data/                      # Data directory
│   ├── raw/                   # Raw data (COCO format)
│   └── processed/             # Processed data (CSV)
│
├── models/                    # Trained models
│   ├── model_mlp.pkl
│   └── scaler.pkl
│
├── src/                       # Source code
│   ├── data_processing/       # Data processing scripts
│   │   └── mediapipe_converter.py
│   ├── training/             # Training scripts
│   │   ├── train_mlp.py
│   │   └── train_gnb.py
│   └── utils/                # Utilities
│       └── vn_accent_restore.py
│
├── apps/                      # Applications
│   ├── webcam_app.py         # Basic webcam app
│   └── pipeline_app.py       # Pipeline app with accent restoration
│
└── README.md
```

## 🔧 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/nhu220840/vn-accent-restore.git
cd vn-accent-restore
```

### Step 2: Create virtual environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

### Data Format

Data should be organized in COCO format:

```
data/raw/data_mini_app_01/
├── train/
│   ├── _annotations.coco.json
│   └── [image files .jpg/.png]
├── valid/
│   ├── _annotations.coco.json
│   └── [image files .jpg/.png]
└── test/
    ├── _annotations.coco.json
    └── [image files .jpg/.png]
```

**Note:**
- JSON file must be named exactly: `_annotations.coco.json`
- Each image must have an annotation in the JSON file
- Label 'DD' will be automatically filtered during processing

## 🔄 Data Preprocessing

Run the preprocessing script to extract hand landmarks:

```bash
python src/data_processing/mediapipe_converter.py
```

**What it does:**
- Extracts hand landmarks using MediaPipe
- Applies data augmentation to training set (9 copies per image)
- Filters out 'DD' label
- Saves results as CSV files

**Output:**
- `data/processed/train_landmarks_augmented.csv`
- `data/processed/valid_landmarks.csv`
- `data/processed/test_landmarks.csv`

**Customization:**
Edit `N_AUGMENTATIONS_PER_IMAGE = 9` in `src/data_processing/mediapipe_converter.py` to change augmentation count.

## 🎓 Model Training

Train the MLP model with GridSearchCV:

```bash
python src/training/train_mlp.py
```

**What it does:**
- Loads processed data
- Normalizes data with StandardScaler
- Searches for best hyperparameters using GridSearchCV
- Evaluates on validation and test sets
- Saves model and scaler

**Output:**
- `models/model_mlp.pkl` - Trained MLP model
- `models/scaler.pkl` - Fitted StandardScaler

**Hyperparameters searched:**
- `hidden_layer_sizes`: [(128, 64), (256, 128, 64)]
- `activation`: ['relu']
- `solver`: ['adam']
- `alpha`: [1e-4, 1e-3]
- `learning_rate_init`: [0.001, 0.0005]

**Note:** Training may take several minutes to hours depending on data size.

## 🚀 Running Applications

### Basic Webcam App

```bash
python apps/webcam_app.py
```

**Features:**
- Real-time hand gesture recognition via webcam
- Displays probability for each class
- Automatically adds gesture to text when held for 1.75 seconds
- Automatically adds space when no hand detected for 2.75 seconds

**Controls:**
- `q` - Quit application
- `s` - Save current word to history
- `c` - Clear current word

### Pipeline App (with Accent Restoration)

```bash
python apps/pipeline_app.py
```

**Features:**
- All features from webcam app
- **Vietnamese accent restoration** (press 'f')
- Displays both raw text and accented text

**Controls:**
- `q` - Quit application
- `f` - Add accents to current text
- `c` - Clear current text

**Note:** Accent restoration uses DistilBERT model and may take a few seconds to process.

## 📝 Complete Workflow

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (place in data/raw/data_mini_app_01/)

# 3. Preprocess data
python src/data_processing/mediapipe_converter.py

# 4. Train model
python src/training/train_mlp.py

# 5. Run application
python apps/webcam_app.py
# or
python apps/pipeline_app.py
```
<!-- ## 📄 License

MIT License. See [LICENSE](LICENSE) for details. -->

<!-- --- -->

## Contact

For questions or contributions, please open an issue or contact:

- 📧 Email: [gianhuw.work@gmail.com](mailto:gianhuw.work@gmail.com)
- 💻 GitHub: [nhu220840](https://github.com/nhu220840)
