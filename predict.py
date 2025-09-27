import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessor import preprocess_text

# --- Constants (Must match those in train.py) ---
MAX_LEN = 200
SAVE_PATH = 'saved_assets'
MODEL_FILE = os.path.join(SAVE_PATH, 'sentimental_model_lstm.h5')
TOKENIZER_FILE = os.path.join(SAVE_PATH, 'tokenizer.pkl')
LABEL_ENCODER_FILE = os.path.join(SAVE_PATH, 'label_encoder.pkl')

# --- Global Assets ---
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(LABEL_ENCODER_FILE, 'rb') as handle:
        label_encoder = pickle.load(handle)
except Exception as e:
    print(f"Error loading assets. Please run train.py first. Error: {e}")
    model = None
    tokenizer = None
    label_encoder = None

def predict_sentiment(raw_text: str):
    """
    Cleans, tokenizes, and predicts the sentiment of a custom text string.
    """
    if model is None or tokenizer is None or label_encoder is None:
        return "Model, Tokenizer, or Label Encoder not loaded. Cannot predict."

    # 1. Clean the raw text
    cleaned_text = preprocess_text(raw_text)
    
    # 2. Convert to integer sequence using the fitted tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Pad the sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )
    
    # 4. Make the prediction (returns probabilities for each class)
    probabilities = model.predict(padded_sequence)[0] 
    
    # 5. Get the predicted class index
    predicted_class_index = np.argmax(probabilities)
    
    # 6. Decode the numerical index back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    print(f"Prediction Probabilities: {probabilities}")
    print(f"Prediction class index: {predicted_class_index}")
    print(f"Predicted Sentiment: {predicted_label}")
    print("----------------------------------")
    
    return predicted_label, probabilities

if __name__ == '__main__':
    sentiment = input("Enter Review : ")
    predict_sentiment(sentiment)
