import streamlit as st
from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('/content/model_final.h5')
tokenizer = joblib.load('/content/tokenizer.pkl')

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequences)
    positive_prob = prediction[0][0] * 100
    negative_prob = (1 - prediction[0][0]) * 100
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    emoji = "😊" if sentiment == 'positive' else "😞"
    return positive_prob, negative_prob, sentiment, emoji

st.title("Sentiment Analysis")
st.write("Enter a review to see the sentiment analysis.")

review = st.text_area("Review", "This is a great product and  i am glad to use it!")

if st.button("Analyze"):
    positive_prob, negative_prob, sentiment, emoji = predictive_system(review)
    
    st.write(f"**Positive:** {positive_prob:.2f}%")
    st.write(f"**Negative:** {negative_prob:.2f}%")
    st.write(f"**Overall Sentiment:** {sentiment}")
    st.write(f"**Emoji:** {emoji}")

