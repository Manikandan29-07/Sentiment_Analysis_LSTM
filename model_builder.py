from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def built_lstm_model(vocab_size: int, max_len: int, embedding_dim: int = 128):
    """
    Defines and compiles the LSTM-based sentiment classification model for multi-class.

    Args:
        vocab_size (int): The maximum number of words in the vocabulary.
        max_len (int): The fixed length of the input sequences (reviews).
        embedding_dim (int): The size of the dense vector for each word.
        num_classes (int): The number of output classes (e.g., 2 for binary, 3 for multi-class).
    
    Returns:
        A compiled Keras Sequential model.
    """
    
    model = Sequential()
    
    # 1. Embedding Layer: Converts integer sequence to dense vectors
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len
    ))
    
    # 2. LSTM Layer: The core recurrent layer for sequence processing
    model.add(LSTM(64))
    
    # 3. Optional Dropout: Reduces overfitting
    model.add(Dropout(0.5))
    
    # 4. Output Layer: Uses 'sigmoid' for binary-class probability distribution
    model.add(Dense(2, activation='sigmoid'))
    
    # 5. Compile the model with 'categorical_crossentropy' for OHE labels
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
