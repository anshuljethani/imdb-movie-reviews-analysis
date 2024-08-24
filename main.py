import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import time

word_index = imdb.get_word_index()


reverse_word_index = {value: key for key, value in word_index.items()}


model = load_model('simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences ([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment= 'Positive' if prediction[0][0] > 0.45220 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
st.markdown("""
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #f5f5f5, #e0e0e0);
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #1e90ff;  /* Vibrant blue */
        text-align: center;
        margin-top: 50px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .subtitle {
        font-size: 24px;
        color: #555;  /* Dark gray */
        text-align: center;
        margin-bottom: 30px;
    }
    .result {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .positive {
        color: #27ae60;  /* Green */
        border: 3px solid #2ecc71;
    }
    .negative {
        color: #e74c3c;  /* Red */
        border: 3px solid #c0392b;
    }
    .blink {
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .fade-in {
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a movie review to classify it as positive or negative.</div>', unsafe_allow_html=True)

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    with st.spinner('Classifying...'):
        time.sleep(2)  # Simulate some processing time
        
        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.45220 else 'Negative'

    # Display the result with advanced styling and animations
    if sentiment == 'Positive':
        st.markdown(f'<div class="result positive fade-in">🎉 {sentiment}</div>', unsafe_allow_html=True)
        st.balloons()  # Display balloons animation for positive sentiment
    else:
        st.markdown(f'<div class="result negative fade-in">💔 {sentiment}</div>', unsafe_allow_html=True)
        st.snow()  # Display snow animation for negative sentiment

else:
    st.write('Please enter a movie review.')