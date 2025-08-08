# 🌾 Smart Crop Advisor – AI-Based Crop Prediction App

A machine learning-powered web application that recommends the **most suitable crop** based on soil nutrients, temperature, humidity, pH, and rainfall. Built using **Streamlit** for an interactive, user-friendly experience.

---

## 🚀 Live Demo

🌐 [Launch the App on Streamlit](https://smart-crop-advisor.streamlit.app/)  
📁 [GitHub Repo](https://github.com/smartswagvivek/Crop-prediction-)

---

## 📦 Features

- ✅ Upload your own agricultural dataset (CSV)
- 🧠 Real-time crop prediction using trained ML model
- 📊 Visualizations: heatmap, feature importance, confusion matrix
- ✍️ Manual data input (N, P, K, pH, etc.)
- 📥 Downloadable prediction reports
- 📈 Model performance shown after training

---

## 🧪 Tech Stack

| Component       | Tools Used           |
|----------------|----------------------|
| 💻 Frontend     | Streamlit            |
| 📦 Backend      | Python, Scikit-learn |
| 📉 ML Model     | RandomForestClassifier |
| 📊 Visualization| Matplotlib, Seaborn  |
| 📄 Data Handling| Pandas, NumPy        |
| 🖼️ Images       | Pillow (PIL)         |

---

## 📂 Sample CSV Format

```csv
N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.8,82.0,6.5,200.0,rice
85,58,41,21.3,80.7,7.0,180.0,maize
76,35,45,25.5,85.0,6.2,250.0,banana
```
#EXTRA
# Clone the repo
git clone https://github.com/smartswagvivek/Crop-prediction-.git
cd Crop-prediction-

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


📦 Crop-prediction-
├── app.py                 # Streamlit app
├── model.pkl              # ML model
├── requirements.txt       # Python dependencies
├── packages.txt           # (optional) for Streamlit Cloud
├── app.yaml               # (optional) deployment config
├── data/                  # Sample CSV files
├── images/                # Screenshots
└── README.md              # Docs
