
# 🇨🇦 Canadian Credit Score Tier Predictor

This is a Streamlit app that predicts a user's Canadian credit score **tier** (e.g. Poor, Fair, Good, Very Good, Excellent) based on their financial inputs using a trained machine learning model.

### 🚀 Try It Live:
👉 [Launch App on Streamlit Cloud](https://YOUR-APP-URL-HERE)

---

## 📂 Features
- Upload a CSV file with financial data
- Get ML-predicted credit tier using a Random Forest Classifier
- View input + predictions with color-coded visuals
- Fully styled with custom Streamlit theming

---

## 📄 Expected CSV Format
Upload a file with these columns:
```
income, payment_history_percent, debt_amount, credit_history_years, utilization_percent, hard_inquiries
```

---

## 🧠 Tech Stack
- Python
- Streamlit
- scikit-learn (ML model)
- pandas (data processing)
- Custom CSS for styling

---

## 💻 Local Run

```bash
git clone https://github.com/akshavib/credit-score-dashboard.git
cd credit-score-dashboard
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 📦 Model Info
- Model: RandomForestClassifier
- Labels: Encoded with LabelEncoder and saved via `pickle`
- Training data: synthetic but realistic, based on Canadian credit scoring standards

---

## 👤 Author
Made by **Akshavi Baskaran** ✨  


