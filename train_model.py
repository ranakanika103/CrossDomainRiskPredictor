import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from utils.preprocessing import clean_text

# 1️⃣ Create model folder
if not os.path.exists("model"):
    os.makedirs("model")

# 2️⃣ Risk mapping
risk_map = {"Low":0, "Medium":1, "High":2, "Critical":3}

# 3️⃣ Base complaints for each class
complaints = {
    "Low": [
        "Minor billing discrepancy in my last invoice",
        "Product slightly delayed but works fine",
        "Received wrong color variant but usable",
        "Packaging slightly damaged but item okay",
        "Small issue with subscription, not urgent"
    ],
    "Medium": [
        "Customer care did not respond to repeated emails",
        "Internet connection unstable for 2 days",
        "Product warranty activation has problem",
        "Received incomplete items, need assistance",
        "Call center repeatedly disconnected me"
    ],
    "High": [
        "Charged multiple times for same product",
        "Delivery executive was rude and unprofessional",
        "Product stopped working within few days",
        "Customer support refused to resolve issue repeatedly",
        "Account temporarily blocked causing inconvenience"
    ],
    "Critical": [
        "Product caused injury while using, urgent medical attention needed",
        "Company violated legal terms and sent misleading invoices",
        "Product unsafe for children, withdraw immediately",
        "Sensitive personal data leaked, urgent action required",
        "Major safety hazard in product, recall immediately"
    ]
}

# 4️⃣ Noise sentences for realism
noise_sentences = [
    " please resolve soon", " yeh unacceptable hai", " I will take legal action",
    " bahut disappointed hoon", " immediately action lo", ""
]

# 5️⃣ Generate 500 examples per risk
data = []
for risk, texts in complaints.items():
    for _ in range(100):  # 5 base texts * 100 = 500 per risk
        text = random.choice(texts)
        noise = random.choice(noise_sentences)
        text_clean = clean_text(text + noise)
        data.append([text_clean, risk_map[risk]])

# Shuffle
random.shuffle(data)

# 6️⃣ DataFrame
df = pd.DataFrame(data, columns=["complaint_text","risk_encoded"])

# Save dataset
df.to_csv("data/multilingual_complaints_balanced.csv", index=False)

# 7️⃣ TF-IDF
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,3))
X = tfidf.fit_transform(df["complaint_text"])
y = df["risk_encoded"]

# 8️⃣ RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# 9️⃣ Save model & TF-IDF
with open("model/risk_model.pkl","wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model, TF-IDF, and dataset saved successfully!")
