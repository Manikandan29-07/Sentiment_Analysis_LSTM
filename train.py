import os 
import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessor import prepare_data
from model_builder import built_lstm_model

FILE_PATH = 'IMDB Dataset.csv'
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
SAVE_PATH = 'saved_assets'
# Note: NUM_CLASSES is now dynamically determined based on the dataset.

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Dataset file '{FILE_PATH}' not found. Please ensure it's in the root directory.")
        return

    df = pd.read_csv(FILE_PATH)
    
    # --- 1. Preprocess data (returns OHE labels in 'y') ---
    X, y, tokenizer, label_encoder = prepare_data(
                            df, 
                            text_column='review', 
                            label_column='sentiment', 
                            max_words=MAX_WORDS, 
                            max_len=MAX_LEN
                        )
    
    # DEBUG: Check the shape of the labels immediately after preprocessing
    print(f"\nLabels 'y' shape received from prepare_data: {y.shape}")
    
    # --- 2. Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # --- 3. Build Model ---
    # Use the second dimension of the OHE array for the correct number of classes
    num_classes_runtime = y_train.shape[1] 
    
    print(f"\nBuilding and compiling multi-class LSTM model with {num_classes_runtime} classes")
    
    model = built_lstm_model(
        vocab_size=MAX_WORDS,
        max_len=MAX_LEN,
        embedding_dim=EMBEDDING_DIM,
        # num_classes=num_classes_runtime # Use the dynamically determined number of classes
    )
    
    # --- 4. Train Model ---
    print("\n________MODEL TRAINING STARTED________")
    model.fit(
        X_train,
        y_train,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        validation_split = 0.15, 
        verbose = 1
    )
    
    # --- 5. Evaluate and Save ---
    print("\nEVALUATING ON TEST DATA")
    loss,accuracy = model.evaluate(X_test, y_test, verbose = 0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    model_saved_path = os.path.join(SAVE_PATH,'sentimental_model_lstm.h5')
    model.save(model_saved_path)
    print(f"Model saved to {model_saved_path}")
    
    tokenizer_file = os.path.join(SAVE_PATH, "tokenizer.pkl")
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_file}")
    
    label_encoder_file = os.path.join(SAVE_PATH, "label_encoder.pkl")
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Label Encoder saved to {label_encoder_file}")
    
if __name__ == '__main__':
    main()
