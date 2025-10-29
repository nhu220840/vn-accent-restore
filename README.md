py -3.12 -m venv .venv 
source .venv/Scripts/activate
pip install --timeout=300 "numpy<2"
pip install --timeout=300 mediapipe pandas
pip install --timeout=300 "opencv-python<4.10"
pip install --timeout=300 "scikit-learn<1.5" "scipy<1.13"
pip install --timeout=300 "seaborn==0.12.2"

