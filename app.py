import streamlit as st
import json
import random
import nltk
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load model and data
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Function to clean up user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# Convert input into a format for the model
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict response
def predict_class(sentence):
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0] if results else None

# Get chatbot response
def get_response(intents_list):
    tag = classes[intents_list] if intents_list is not None else "noanswer"
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

# Streamlit UI with farming theme
st.set_page_config(page_title="Farming Helper Chatbot", page_icon="🌾")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("farming.png", width=200)

st.title("🌾 Farming Helper Chatbot 🤖")
st.write("Welcome! Ask me anything about farming and agricultural equipment.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text_area("👨‍🌾 You:", message["content"], height=100, disabled=True)

    else:
        st.success("🤖 Chatbot:")
        st.info(message["content"])

# User input
user_input = st.text_input("👨‍🌾 You:", "", placeholder="Type your farming-related query here...")

# Process response
if st.button("🚜 Ask Chatbot") and user_input:
    chatbot_response = get_response(predict_class(user_input))
    
    # Store user query and chatbot response
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "bot", "content": chatbot_response})
    
    # Refresh the page to display the conversation history
    st.rerun()
