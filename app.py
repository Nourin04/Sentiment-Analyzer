import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os

# Function to download the model files from Google Drive
def download_file_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

# File IDs from Google Drive
MODEL_FILE_ID = '1lZ5_ctzJXSFwa0FZZMBJHF8HF6mM3jgB'
TOKENIZER_FILE_ID = '1QaILoCOEbewXPlQXZ6HCXhOMo6ekNBho'
ENCODER_FILE_ID = '1PHSatOkcudmY6RHTzCKqQqB7Zr-JX-v1'

# File paths
model_path = "emotion_detection_model.keras"
tokenizer_path = "tokenizer.pkl"
encoder_path = "label_encoder.pkl"

# Download files if they don't exist
download_file_from_gdrive(MODEL_FILE_ID, model_path)
download_file_from_gdrive(TOKENIZER_FILE_ID, tokenizer_path)
download_file_from_gdrive(ENCODER_FILE_ID, encoder_path)

# Load model, tokenizer, and label encoder
model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Dynamically get max_length from the model input shape
max_length = model.input_shape[1]

# Prediction function
def predict_emotion(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    
    # Check if the input is valid
    if len(input_sequence[0]) == 0:
        return "Input text contains unknown words or is empty."
    
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label[0]

# Streamlit app UI
st.title("😊 Real-Time Emotion Detection App")
st.write("Enter any text below, and I'll predict its emotion!")

# Input text box
user_input = st.text_input("Type your message here:")

# Prediction button
if st.button("Predict Emotion"):
    if user_input.strip() != "":
        result = predict_emotion(user_input)
        if result == "Input text contains unknown words or is empty.":
            st.error("The input text contains unknown words. Please try with different text.")
        else:
            st.success(f"The predicted emotion is: **{result}**")
    else:
        st.error("Please enter some text to get a prediction.")

# Add a footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit and TensorFlow")
