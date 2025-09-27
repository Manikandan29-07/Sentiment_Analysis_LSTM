import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np # Added numpy

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
    
# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text:str):
    """
    Cleans and preprocesses a single text string.
    """
    text = text.lower()
    text = re.sub(r'<.*?>','', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = text.split()
    preprocessed_word = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(preprocessed_word)


def tokenize_and_pad(texts, max_words, max_len):
    """
    Fits Keras Tokenizer, converts texts to sequences, and pads them.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences,
        maxlen = max_len,
        padding = 'post',
        truncating = 'post'
    )
    return padded_sequences, tokenizer


def prepare_data(df:pd.DataFrame, text_column:str, label_column:str, max_words:int, max_len:int):
    """
    Applies ALL preprocessing steps: Cleaning, Tokenization, Padding, and One-Hot Encoding.
    """
    print("Data Preprocessing Started (Multi-Class)")
    
    # 1. Custom Text Cleaning
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    
    # 2. Label Encoding (Numerical mapping)
    le = LabelEncoder()
    # First, convert text labels to numerical indices (e.g., Negative:0, Positive:1)
    numerical_labels = le.fit_transform(df[label_column]) 

    # 3. One-Hot Encoding (OHE) - This ensures the output is 2D
    y = to_categorical(numerical_labels, num_classes=len(le.classes_))
    
    # DEBUG: Crucial check to confirm y is 2D before returning
    print(f"DEBUG: OHE labels 'y' shape after to_categorical: {y.shape}")

    # 4. Tokenization and Padding
    X, tokenizer = tokenize_and_pad(df['cleaned_text'].values, max_words, max_len)
    
    print("Data Preprocessing Completed")
    
    # Return OHE labels (y) which is a 2D array
    return X, y, tokenizer, le
