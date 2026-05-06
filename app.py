import streamlit as st
import pickle

# Load
model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("Sentiment Analysis (Naive Bayes)")

text = st.text_area("Enter review:")

if st.button("Predict"):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec)[0]
    
    if pred == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")