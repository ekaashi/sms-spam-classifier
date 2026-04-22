import streamlit as st
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---- Page Config ----
st.set_page_config(page_title="Spam Detector AI", page_icon="🚀", layout="centered")

# ---- NLTK setup ----
nltk.download('punkt')
nltk.download('stopwords')

# ---- Load model ----
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ---- Preprocessing ----
ps = PorterStemmer()
sw = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in sw and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---- PREMIUM HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #00C9A7;'>🚀 Spam Detection AI</h1>
    <p style='text-align: center; font-size:18px; color:gray;'>
    Detect spam messages instantly using Machine Learning
    </p>
""", unsafe_allow_html=True)

st.write("")

# ---- INPUT CARD ----
st.markdown("""
<div style="
    background-color:#1e1e1e;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);">
""", unsafe_allow_html=True)

input_sms = st.text_area("✉️ Enter your message", height=150)

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---- BUTTON CENTER ----
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_btn = st.button("🔍 Analyze Message")

st.write("")

# ---- PREDICTION ----
if predict_btn:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        with st.spinner("🤖 AI is analyzing your message..."):

            transformed_sms = transform_text(input_sms)
            vector_input = vectorizer.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        st.write("")

        # ---- RESULT UI ----
        if result == 1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ff4d4d, #ff1a1a);
                padding:25px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:24px;
                font-weight:bold;
                box-shadow:0px 5px 20px rgba(255,0,0,0.4);">
                🚨 SPAM DETECTED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #00C9A7, #00b894);
                padding:25px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:24px;
                font-weight:bold;
                box-shadow:0px 5px 20px rgba(0,200,150,0.4);">
                ✅ SAFE MESSAGE
            </div>
            """, unsafe_allow_html=True)

        # ---- CONFIDENCE SCORE ----
        prob = model.predict_proba(vector_input)[0]

        st.write("")
        st.subheader("📊 Confidence Score")

        st.progress(int(max(prob)*100))
        st.write(f"Confidence: **{round(max(prob)*100,2)}%**")

        # ---- DEBUG SECTION ----
        with st.expander("🔍 View Processed Text"):
            st.write(transformed_sms)

# ---- FOOTER ----
st.markdown("""
<hr>
<p style='text-align: center; color: gray;'>
Built with ❤️ using Machine Learning & Streamlit
</p>
""", unsafe_allow_html=True)
