import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")

API_URL = "https://fake-news-api-1-4pzb.onrender.com"

text = st.text_area("Enter news text", height=200)

if st.button("Check News"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        res = requests.post(API_URL, json={"text": text})

        if res.status_code == 200:
            data = res.json()
            st.success(f"Prediction: {data['prediction']}")
            st.info(f"Confidence: {data['confidence']}")
        else:
            st.error("API error")
