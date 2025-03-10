# Import libraries
import tensorflow as tf
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import CustomObjectScope
import gdown
import os

# Function to download the model files from Google Drive
def download_file_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

# File IDs from Google Drive
MODEL_FILE_ID = '1ase6CSuuke-IvZ-IsYQucE5J_i-RnHIi'
TOKENIZER_FILE_ID = '1T9qGlyTFligYzHAE4qsTTD9NfTeT37kP'
ENCODER_FILE_ID = '1yf-8_bA--JRad9q24y7TmJ9F2NEgyatI'

# File paths
model_path = "emotion_detection_model.keras"
tokenizer_path = "tokenizer.pkl"
encoder_path = "label_encoder.pkl"

# Download files if they don't exist
download_file_from_gdrive(MODEL_FILE_ID, model_path)
download_file_from_gdrive(TOKENIZER_FILE_ID, tokenizer_path)
download_file_from_gdrive(ENCODER_FILE_ID, encoder_path)

# Load model, tokenizer, and label encoder
try:
    with CustomObjectScope({'RMSprop': RMSprop}):
        model = load_model(model_path, compile=False)
        model.compile()  # Compile after loading
except AttributeError as e:
    st.error("Model loading failed due to an attribute issue. Try updating TensorFlow or Keras.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading the model: {e}")
    st.stop()

# Load Tokenizer and Encoder
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Dynamically get max_length from the model input shape
max_length = model.input_shape[1]

# Prediction function (without tf.function decorator)
def predict_emotion(input_text):
    # Convert input text to a sequence
    input_sequence = tokenizer.texts_to_sequences([input_text])

    # Check for empty input or unknown words
    if len(input_sequence[0]) == 0:
        return "Input text contains unknown words or is empty."

    # Pad the input sequence
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)

    # Convert to tensor explicitly
    input_tensor = tf.convert_to_tensor(padded_input_sequence, dtype=tf.float32)

    try:
        # Make a prediction
        prediction = model.predict(input_tensor, verbose=0)

        # Check if prediction is valid
        if prediction is not None and prediction.size > 0:
            predicted_label_index = np.argmax(prediction[0])
            predicted_label = label_encoder.inverse_transform([predicted_label_index])
            return predicted_label[0]
        else:
            return "Error: Model prediction failed."
    except Exception as e:
        return f"Prediction Error: {e}"

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
