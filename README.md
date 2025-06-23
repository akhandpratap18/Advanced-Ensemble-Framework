# 🔐 Advanced Ensemble Framework for Defending Against Obfuscated Windows, Android, and IoT Malware

An intelligent and extensible malware detection dashboard that leverages advanced ensemble models to detect obfuscated and non-obfuscated threats across Android, Windows, and IoT-based platforms. This tool offers in-depth analysis, progress visualization, class-wise metrics, and label encoding transparency — all through an interactive Streamlit interface.

---

## 🚀 Features

- **Upload & Analyze**: Upload test CSVs for any malware family and get predictions using pretrained models.
- **Model Variety**: Supports Android, Windows, IoT WiFi, and Obfuscated malware detection.
- **Interactive Dashboard**: Animated gauge, class-wise accuracy bar charts, confusion matrix, and detailed classification metrics.
- **Label Decoder**: Understand the encoded labels for clarity in predictions.
- **Download Results**: Export predictions and correctness metrics in one click.

---

## 📁 Folder Structure
- WeightedModels/ # Trained .joblib models
-  app.py # Main Streamlit dashboard app
-  requirements.txt # Required Python libraries
-  README.md # Project documentation
-  TestDatasets/ # Sample CSVs for quick testing
 

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ensemble-malware-detector.git
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

cd ensemble-malware-detector
