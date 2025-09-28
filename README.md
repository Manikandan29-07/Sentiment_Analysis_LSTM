Deep Learning Sentiment Analysis of Movie Reviews (LSTM) 🎬
Project Overview
This project implements an end-to-end deep learning pipeline for classifying the sentiment of movie reviews. This is an upgrade from traditional machine learning methods, utilizing a Bidirectional Long Short-Term Memory (Bi-LSTM) recurrent neural network implemented with Keras/TensorFlow.

The primary goal of the Bi-LSTM is to capture the sequence and context of words by processing text forwards and backwards, enabling the model to accurately handle complex linguistic structures like negation and subtle sentiment shifts. The model is configured for Binary Classification (Positive/Negative).

Key Features
Bidirectional Sequence Modeling: Uses a Bi-LSTM layer, allowing the model to capture context from both previous and subsequent words, greatly enhancing understanding compared to a standard LSTM.



Keras Tokenization & Padding: Raw text is converted into fixed-length numerical sequences required by the neural network using a dedicated tokenizer and padding strategy.

Modular Deep Learning Architecture: The model definition is separated into a dedicated model_builder.py file, making it easy to adjust hyperparameters and layers.

Robust Preprocessing: Includes lowercasing, HTML tag removal, lemmatization, and stop-word removal, all consolidated in one preparation function.

Asset Preservation: Critical artifacts—the trained Keras model, the fitted Keras Tokenizer, and the Label Encoder—are saved to ensure reproducible and portable predictions.

📂 Project Structure
sentiment-analysis-lstm/
├── IMDB Dataset.csv            # The raw dataset (e.g., review, sentiment)
├── data_preprocessor.py        # Text cleaning, tokenization, padding, and label encoding
├── model_builder.py            # Defines the Keras Embedding -> Bi-LSTM -> Dense architecture
├── train.py                    # **The main orchestrator**: Loads data, trains, evaluates, and saves assets
├── predict.py                  # Script for making live predictions on new text
└── saved_assets/               # Directory to save the trained model artifacts
    ├── sentimental_model_bilstm.h5 # The trained Bi-LSTM Keras model (New Version)
    ├── tokenizer.pkl           # The fitted Keras Tokenizer (maps words to IDs)
    └── label_encoder.pkl       # The fitted Label Encoder (maps 'positive'/'negative' to 1/0)

🚀 How to Run the Project

1. Prerequisites & Environment
   Ensure you have Python installed and create/activate a virtual environment:

# Create and activate a virtual environment

python -m venv venv
source venv/bin/activate

2. Install Libraries
   The dependencies have changed for deep learning:

pip install pandas scikit-learn nltk tensorflow joblib numpy

3. Dataset Setup
   Download your dataset and ensure the file is named IMDB Dataset.csv and placed in the project's root directory.

4. Run the Pipeline (Training)
   The train.py script now handles all steps: preprocessing, model building, training, evaluation, and saving the artifacts.

python train.py

This will start the training across multiple epochs and save all necessary files to the saved_assets/ directory.

Evaluation Results
(NOTE: Run python train.py and paste the new evaluation results here. The Bi-LSTM model is expected to outperform the previous Logistic Regression model.)

Test Accuracy:
INSERTNEWLSTMACCURACYHERE
Classification Report
[INSERT NEW CLASSIFICATION REPORT HERE]

Make a Live Prediction
The predict.py script loads the saved model and tokenizer to test new sentences. This is where you can test sentences like "That was not really fantastic" to see the Bi-LSTM's ability to handle negation.

python predict.py

Modify the test reviews in the if **name** == "**main**": block of predict.py to test your own custom reviews.
