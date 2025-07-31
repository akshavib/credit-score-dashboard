# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv('data/canadian_credit_score_data.csv')

# Features and target
X = df.drop('credit_score_category', axis=1)
y = df['credit_score_category']

# Encode target labels (e.g., 'Poor' → 0)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Model and label encoder saved successfully!")
