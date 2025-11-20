import streamlit as st
import pickle
import re

# -------------------------
# Load Model & Vectorizer
# -------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("emoji_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# Clean text function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.title("😊 Emoji Prediction App")

st.write("Type some text and the model will predict the most suitable emoji.")

# User Input
user_input = st.text_area("Enter your text here:", placeholder=" ")

if st.button("Predict Emoji"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.success(f" Predicted Emoji: **{prediction}**")
    else:
        st.warning("Please enter some text!")
