import streamlit as st
import pickle
import pandas as pd
from utils.preprocessing import clean_text
import numpy as np

# 1️⃣ Load model and TF-IDF
with open("model/risk_model.pkl","rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl","rb") as f:
    tfidf = pickle.load(f)

# 2️⃣ Action mapping
actions = {
    "Low": "Monitor or assign to general support",
    "Medium": "Assign to support team for follow-up",
    "High": "Immediate team attention required",
    "Critical": "Escalate to legal/safety/compliance team"
}

# 3️⃣ Streamlit UI
st.title("📌 Complaint Risk Classification App")
st.write("Enter a complaint (English/Hinglish/Hindi) and get risk level prediction.")

user_input = st.text_area("Enter complaint here:")

if st.button("Predict Risk"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a complaint text!")
    else:
        # Clean & vectorize
        cleaned = clean_text(user_input)
        vect = tfidf.transform([cleaned])

        # Predict
        pred = model.predict(vect)[0]
        label_map = {0:"Low", 1:"Medium", 2:"High", 3:"Critical"}
        pred_label = label_map[pred]

        st.success(f"Predicted Risk Level: {pred_label}")
        st.info(f"Recommended Action: {actions[pred_label]}")

        # Top contributing words
        vect_array = vect.toarray()[0]
        top_indices = vect_array.argsort()[-5:][::-1]
        feature_names = tfidf.get_feature_names_out()
        top_features = [feature_names[i] for i in top_indices]

        st.write("Top words contributing to risk prediction:")
        st.write(top_features)
