import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Config
st.set_page_config(
    page_title="🌾 Smart Crop Advisor", 
    layout="wide",
    page_icon="🌱"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #f0f0f0;
    }
    .st-ae {
        background-color: #e0e0e0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title Section
st.title("🌾 Smart Crop Advisor")
st.markdown("""
    <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; margin-bottom:20px;'>
        <h3 style='color:#2e7d32;'>Optimize Your Farming Decisions with AI</h3>
        <p>Upload your dataset and input environmental conditions to get science-based crop recommendations tailored to your land.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3079/3079165.png", width=100)
    st.title("Settings")
    
    # Theme Toggle
    theme = st.radio("Color Theme", ["🌞 Light", "🌙 Dark"], index=0)
    if theme == "🌙 Dark":
        st.markdown("""<style>.main {background-color: #0e1117; color: white;}</style>""", 
                  unsafe_allow_html=True)
    
    # Model Parameters
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
    test_size = st.slider("Test Size (%)", 10, 40, 25, 5)
    
    # About Section
    st.markdown("---")
    st.subheader("About")
    st.info("""
        This AI-powered system recommends the most suitable crops based on:
        - Soil nutrients (N, P, K)
        - Weather conditions
        - pH levels
        - Rainfall patterns
    """)

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload Your Agricultural Dataset (CSV format)", 
                                type=["csv"],
                                help="Please upload a CSV file with columns for N, P, K, temperature, humidity, ph, rainfall, and crop label")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Data Preview Section
    with st.expander("🔍 Dataset Preview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 5 Rows")
            st.dataframe(df.head().style.highlight_max(axis=0, color='#e6f3ff'))
        with col2:
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe().style.background_gradient(cmap='YlGn'))
    
    # Prepare data
    X = df.drop('label', axis=1)
    y = df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size/100, 
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Input Section with Tabs
    st.markdown("---")
    st.header("🌦️ Environmental Conditions Input")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Upload"])
    
    with tab1:
        cols = st.columns(4)
        with cols[0]:
            n = st.slider("Nitrogen (N) ppm", 0, 140, 90, 
                         help="Nitrogen level in parts per million")
            p = st.slider("Phosphorus (P) ppm", 5, 145, 42,
                         help="Phosphorus level in parts per million")
        with cols[1]:
            k = st.slider("Potassium (K) ppm", 5, 205, 43,
                         help="Potassium level in parts per million")
            temperature = st.slider("Temperature (°C)", 8.0, 45.0, 20.8, 0.1,
                                   help="Average daily temperature")
        with cols[2]:
            humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, 0.1,
                               help="Relative humidity percentage")
            ph = st.slider("pH Level", 3.5, 9.5, 6.5, 0.1,
                          help="Soil pH level (acidity/alkalinity)")
        with cols[3]:
            rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 200.0, 1.0,
                                help="Annual rainfall in millimeters")
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Predict Optimal Crop", use_container_width=True)
    
    with tab2:
        st.warning("Batch prediction feature coming soon!")
        batch_file = st.file_uploader("Upload batch data (CSV)", type=["csv"])
    
    if predict_btn:
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        predicted_crop = le.inverse_transform(prediction)[0]
        
        # Prediction Result Section
        st.markdown("---")
        st.header("🎯 Recommendation Result")
        
        # Create a nice card for the prediction
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; border-left:6px solid #4CAF50;'>
            <h2 style='color:#2e7d32;'>Recommended Crop: <span style='color:#1b5e20;'>{predicted_crop.capitalize()}</span></h2>
            <p>Based on your input conditions, this crop has the highest suitability for your land.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics and Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Model Accuracy
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy Score", f"{acc * 100:.2f}%", 
                    help="The percentage of correct predictions on test data")
            
            # Feature Importance
            st.subheader("🔝 Key Influencing Factors")
            importances = model.feature_importances_
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=importances, y=X.columns, palette="viridis", ax=ax1)
            ax1.set_title("Feature Importance Ranking")
            st.pyplot(fig1)
            
        with col2:
            # Crop Info & Image
            crop_info = {
                "rice": {
                    "desc": "🌾 Rice thrives in warm, humid conditions with water-logged fields. Ideal temperature range: 20-35°C. Requires high nitrogen.",
                    "img": "https://images.unsplash.com/photo-1604977046806-87b4e8a8e1a8",
                    "tips": ["Requires standing water during growth", "Needs clayey or loamy soil", "pH range: 5.0-7.5"]
                },
                "maize": {
                    "desc": "🌽 Maize grows best in warm climates with well-drained soil. Optimal temperature: 18-27°C. Sensitive to frost.",
                    "img": "https://images.unsplash.com/photo-1601593768799-76e9c4288e0b",
                    "tips": ["Plant in blocks for better pollination", "Needs moderate phosphorus", "pH range: 5.8-7.0"]
                },
                "banana": {
                    "desc": "🍌 Banana plants require tropical conditions with high humidity. Temperature range: 15-35°C. Needs protection from wind.",
                    "img": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e",
                    "tips": ["Requires rich, well-drained soil", "Needs high potassium", "pH range: 5.5-7.0"]
                }
            }
            
            info = crop_info.get(predicted_crop, {
                "desc": "ℹ️ General farming advice: Rotate crops regularly to maintain soil health.",
                "img": "https://images.unsplash.com/photo-1464226184884-fa280b87c399",
                "tips": ["Test soil regularly", "Monitor weather patterns", "Consult local agricultural experts"]
            })
            
            st.subheader(f"📚 About {predicted_crop.capitalize()}")
            st.info(info["desc"])
            
            # Display image with rounded corners
            st.image(info["img"], 
                   caption=f"{predicted_crop.capitalize()} Cultivation", 
                   use_column_width=True)
            
            # Growing tips
            st.subheader("🌿 Growing Tips")
            for tip in info["tips"]:
                st.markdown(f"- {tip}")
        
        # Additional Analytics
        st.markdown("---")
        st.header("📊 Advanced Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Confusion Matrix", "Data Distribution"])
        
        with tab1:
            st.subheader("Feature Correlation Heatmap")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)
            st.caption("Understand how different environmental factors relate to each other")
        
        with tab2:
            st.subheader("Model Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax3)
            ax3.set_xlabel("Predicted Crops")
            ax3.set_ylabel("Actual Crops")
            st.pyplot(fig3)
            st.caption("See how often the model confuses one crop for another")
        
        with tab3:
            st.subheader("Feature Distribution")
            selected_col = st.selectbox("Select feature to visualize", X.columns)
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            sns.histplot(df[selected_col], kde=True, ax=ax4)
            ax4.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig4)
        
        # Download Section
        st.markdown("---")
        st.header("📥 Export Results")
        
        if st.button("💾 Download Prediction Report"):
            output = io.StringIO()
            output.write(f"SMART CROP ADVISOR REPORT\n{'='*30}\n\n")
            output.write(f"Recommended Crop: {predicted_crop}\n\n")
            output.write("Input Parameters:\n")
            output.write(f"- Nitrogen (N): {n} ppm\n")
            output.write(f"- Phosphorus (P): {p} ppm\n")
            output.write(f"- Potassium (K): {k} ppm\n")
            output.write(f"- Temperature: {temperature}°C\n")
            output.write(f"- Humidity: {humidity}%\n")
            output.write(f"- pH Level: {ph}\n")
            output.write(f"- Rainfall: {rainfall} mm\n\n")
            output.write(f"Model Accuracy: {acc*100:.2f}%\n")
            
            st.download_button(
                label="⬇️ Download Full Report",
                data=output.getvalue(),
                file_name=f"crop_recommendation_{predicted_crop}.txt",
                mime="text/plain",
                help="Download a detailed report of your crop recommendation"
            )

else:
    # Welcome screen with sample data option
    st.warning("Please upload your agricultural dataset to get started")
    
    # Sample data section
    with st.expander("🧪 Don't have data? Try with our sample", expanded=False):
        st.info("""
            You can test the system with this sample data format:
            
            | N  | P  | K  | temperature | humidity | ph    | rainfall | label   |
            |----|----|----|-------------|----------|-------|----------|---------|
            | 90 | 42 | 43 | 20.8        | 82.0     | 6.5   | 200.0    | rice    |
            | 85 | 58 | 41 | 21.3        | 80.7     | 7.0   | 180.0    | maize   |
            | 76 | 35 | 45 | 25.5        | 85.0     | 6.2   | 250.0    | banana  |
            
            [Download sample CSV template](#)
        """)
    
    # Features highlights
    st.markdown("---")
    st.header("✨ Key Features")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; height:180px;'>
            <h4>🌡️ Climate Analysis</h4>
            <p>Understand how temperature and humidity affect crop suitability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; height:180px;'>
            <h4>🌱 Soil Health</h4>
            <p>Optimize N, P, K levels and pH balance for maximum yield</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div style='background-color:#f3e5f5; padding:15px; border-radius:10px; height:180px;'>
            <h4>📈 Data-Driven</h4>
            <p>Machine learning models trained on agricultural research data</p>
        </div>
        """, unsafe_allow_html=True)