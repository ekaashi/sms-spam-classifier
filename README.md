# sms-spam-classifier
# 📩 SMS Spam Classifier (Machine Learning + Streamlit)

A lightweight and efficient **SMS Spam Detection Web App** built using Machine Learning and deployed with Streamlit. The application classifies messages as **Spam 🚨** or **Not Spam ✅** in real-time.

---

## 🚀 Features

* Real-time SMS classification
* Clean and simple UI using Streamlit
* Text preprocessing (lowercasing, tokenization, stopword removal, stemming)
* TF-IDF vectorization for feature extraction
* Trained Machine Learning model for accurate prediction
* Fast and responsive deployment

---

## 🧠 Tech Stack

* **Python**
* **Scikit-learn**
* **Pandas & NumPy**
* **Streamlit**
* **NLTK (for text preprocessing)**

---

## ⚙️ How It Works

1. User enters an SMS message
2. Text is preprocessed:

   * Lowercasing
   * Tokenization
   * Stopword removal
   * Stemming
3. TF-IDF converts text into numerical features
4. Trained model predicts whether the message is spam or not

---

## 📂 Project Structure

```
sms-spam-classifier/
│
├── app.py                # Streamlit app
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt     # Dependencies
├── sms_spam_detection.ipynb  # Model training notebook
```

---

## 🛠️ Installation & Run Locally

```bash
git clone <repo-link>
cd sms-spam-classifier
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The app is deployed using **Streamlit Cloud**, making it accessible from anywhere.

---

## 📌 Future Improvements

* Improve model accuracy using advanced algorithms
* Add deep learning (LSTM / BERT)
* Enhance UI/UX
* Remove NLTK dependency for faster deployment

---

## 🎯 Purpose

This project demonstrates practical implementation of:

* Text preprocessing
* Feature engineering
* Machine learning deployment

It is suitable for learning and showcasing **end-to-end ML pipelines**.

---

## 👨‍💻 Author

Abhishek
B.Tech (AI & DS) Student
