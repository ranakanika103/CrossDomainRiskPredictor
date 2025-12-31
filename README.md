# 🚀 Cross-Domain Complaint Risk Predictor

A machine learning–based system to **predict complaint risk across multiple domains and languages** using **TF-IDF vectorization and classical ML models**.
---

## 🔹 Project Overview

Organizations receive customer complaints from different domains and in multiple languages.  
Manually identifying **high-risk complaints** is time-consuming and error-prone.

This project solves that problem by:
- Transforming complaint text into numerical features using **TF-IDF**
- Training a **machine learning risk prediction model**
- Supporting **cross-domain and multilingual complaints**
- Providing a **deployable application (`app.py`)**

🎯 **Goal:** Predict whether a complaint is **high-risk or low-risk** with strong generalization and minimal overfitting.

---

## ✨ Key Highlights

- ✅ **TF-IDF–based text representation**
- ✅ **Cross-domain complaint handling**
- ✅ **Multilingual complaint datasets**
- ✅ **Balanced training data**
- ✅ **Reusable trained model (`.pkl`)**
- ✅ **Modular preprocessing utilities**
- ✅ **Deployment-ready Python app**

---

## 🛠 Tech Stack

| Tool | Purpose |
|-----|--------|
| Python | Core programming |
| Jupyter Notebook | Model training & experimentation |
| Pandas / NumPy | Data handling |
| Scikit-learn | ML models & TF-IDF |
| Pickle | Model serialization |
| Flask / Streamlit (optional) | App deployment |
| NLP | Text preprocessing & vectorization |

---

## 🔹 Project Structure

CrossDomainRiskProject/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│
├── data/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ ├── multilingual_complaints_balanced/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ ├── multilingual_complaints_dataset/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ └── new_complaints/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│
├── utils/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ ├── pycache/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ │ └── preprocessing.cpython-311.pyc&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ └── preprocessing.py&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│
├── models/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ ├── risk_model.pkl&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│ └── tfidf_vectorizer.pkl&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
│
├── venv_risk/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
├── train_model/&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
├── app.py&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
├── README.md&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>
└── requirements.txt&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br>

---

## 📊 Dataset Description

- **Multilingual complaint text**
- Complaints from **multiple domains**
- Balanced dataset to reduce bias
- New complaint samples supported for real-time prediction

### Preprocessing includes:
- Text cleaning
- Lowercasing
- Stopword removal
- TF-IDF vectorization

---

## ⚙️ Methodology

### 1️⃣ Text Preprocessing
Handled inside `utils/preprocessing.py`:
- Noise removal
- Token normalization
- Language-independent cleaning

---

### 2️⃣ Feature Engineering
- **TF-IDF Vectorizer**
- Converts complaint text into numerical vectors
- Saved as `tfidf_vectorizer.pkl`

---

### 3️⃣ Model Training
- Trained using **Scikit-learn**
- Focus on **generalization**
- Avoided overfitting using:
  - Balanced data
  - Clean feature space
  - Proper evaluation


---

### 4️⃣ Prediction Pipeline
- New complaints loaded from `new_complaints/`
- Same preprocessing + TF-IDF applied
- Risk prediction generated via trained model

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

> Emphasis on **reliable predictions** rather than over-optimized accuracy.

---

## 🔮 Future Improvements

* 🌍 Advanced multilingual embeddings

* 📦 API deployment

* 📊 Interactive dashboard

* ⏱ Real-time complaint streaming

* 🧠 Model explainability (SHAP / LIME)

---

## 👩‍💻 Author

Kanika Rana
💼 Aspiring Data Scientist / ML Engineer

LinkedIn: www.linkedin.com/in/kanika-rana-0681b4372