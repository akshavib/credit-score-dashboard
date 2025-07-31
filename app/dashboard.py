import streamlit as st
import pandas as pd
import pickle

# 📐 Set wide layout
st.set_page_config(page_title="Credit Tier Estimator", layout="wide")

# 🌲 Forest green background and white text styling
st.markdown("""
    <style>
        body, .reportview-container .main, .block-container {
            background-color: #2e4d35;
            color: #ffffff;
        }

        h1, h2, h3, h4, h5, h6, .stMarkdown {
            color: #ffffff !important;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }

        .stFileUploader, .stDataFrame {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 10px;
        }

        .stDataFrame table {
            background-color: white;
            color: black;
        }

        .stInfo {
            background-color: #40664d;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Load model and label encoder
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# 🏷️ Header
st.title("🇨🇦 Canadian Credit Score Tier Predictor")
st.markdown("Upload your financial profile to see your predicted credit score category.")
st.markdown("*Expected columns: `income`, `payment_history_percent`, `debt_amount`, `credit_history_years`, `utilization_percent`, `hard_inquiries`*")

# 📤 Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Input Data")
    st.dataframe(df)

    # Predict and decode
    preds = model.predict(df)
    decoded = encoder.inverse_transform(preds)

    df["Predicted Credit Tier"] = decoded

    # 🌈 Color map
    color_map = {
        "Poor": "background-color: #f8d7da; color: #721c24",
        "Fair": "background-color: #fff3cd; color: #856404",
        "Good": "background-color: #d1ecf1; color: #0c5460",
        "Very Good": "background-color: #d4edda; color: #155724",
        "Excellent": "background-color: #c6f6d5; color: #22543d"
    }

    def highlight(row):
        return [color_map.get(row["Predicted Credit Tier"], "")] * len(row)

    st.subheader("🔮 Predicted Credit Score Tiers")
    st.dataframe(df.style.apply(highlight, axis=1))

else:
    st.info("👆 Please upload a CSV file to get started.")
