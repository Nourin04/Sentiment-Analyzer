

# 😊 Real-Time Emotion Detection App

This is a web application that predicts the **emotion** behind any text input using a deep learning model built with TensorFlow. The app uses a pre-trained model and offers real-time predictions via an interactive Streamlit interface.

---

## 🚀 Features

- 🔍 **Real-Time Emotion Detection** – Predicts emotions like *happy*, *sad*, *angry*, *fear*, etc., from input text.  
- 📦 **Pre-trained Model** – Uses a trained TensorFlow model with a tokenizer and label encoder for accurate predictions.  
- 🌐 **Google Drive Integration** – Automatically downloads the model and necessary files from Google Drive.  
- 💡 **User-Friendly Interface** – Built using Streamlit for an intuitive and responsive UI.  
- ⚡ **Fast & Efficient** – Optimized for quick response times.

---

## 🛠️ Technologies Used

- **Python 3.12+**
- **TensorFlow 2.x**
- **Streamlit** – For building the web app
- **Keras** – For model training and prediction
- **NumPy** – For numerical computations
- **gdown** – To download files from Google Drive
- **Pickle** – For loading tokenizers and label encoders  

---

## 📂 Project Structure

```
real-time-emotion-detector/
│
├── app.py                    # Main Streamlit app script
├── emotion_detection_model.keras  # Pre-trained TensorFlow model (downloaded)
├── tokenizer.pkl             # Tokenizer for text sequences (downloaded)
├── label_encoder.pkl         # Label encoder for output labels (downloaded)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-time-emotion-detector.git
cd real-time-emotion-detector
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Model Files (Automatically)
The app will automatically download:
- `emotion_detection_model.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`  

These files will be downloaded from Google Drive using their unique file IDs.

---

## 🚦 How to Run the App

(https://sentiment-analyzer12.streamlit.app/)

---

## 💻 Usage

1. Enter a sentence or phrase in the text box.  
2. Click the **"Predict Emotion"** button.  
3. The predicted emotion will be displayed below the input field.

---

## 🔧 Troubleshooting

- **Model Loading Error**:  
  Update TensorFlow and Keras to the latest versions.  
  ```bash
  pip install --upgrade tensorflow keras
  ```

- **Unknown Words Issue**:  
  The model uses a predefined tokenizer. If you input words that weren't in the training set, try rephrasing your sentence.

- **File Download Error**:  
  Ensure that the files are shared publicly on Google Drive.

---

## 📝 Requirements

List of essential dependencies (already in `requirements.txt`):

```
tensorflow>=2.9.0
streamlit>=1.10.0
gdown
numpy
pickle5
```

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👨‍💻 Author

- **Noureen AC** – [GitHub](https://github.com/Nourin04) | [LinkedIn](https://linkedin.com/in/noureen-ac)

---

## 🙏 Acknowledgments

- Special thanks to **TensorFlow** and **Streamlit** communities for their valuable tools and documentation.
- Model training was inspired by various open-source sentiment analysis and NLP projects.

---

Let me know if you'd like any additional sections! 🚀
