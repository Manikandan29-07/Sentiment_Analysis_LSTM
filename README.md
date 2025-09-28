# Sentiment Analysis of Movie Reviews 🎬

## Project Overview

This project implements an end-to-end deep learning pipeline for classifying the sentiment of movie reviews. This is an upgrade from traditional machine learning methods, utilizing a **Bidirectional Long Short-Term Memory (Bi-LSTM)** recurrent neural network implemented with Keras/TensorFlow.(Previously we have tried LSTM)

The primary goal of the Bi-LSTM is to capture the **sequence and context** of words by processing text forwards and backwards, enabling the model to accurately handle complex linguistic structures like negation and subtle sentiment shifts. The model is configured for **Binary Classification** (Positive/Negative).
## Key Features

* **Bidirectional Sequence Modeling**: Uses a Bi-LSTM layer, allowing the model to capture context from both previous and subsequent words, greatly enhancing understanding compared to a standard LSTM.
  

* **Keras Tokenization & Padding**: Raw text is converted into fixed-length numerical sequences required by the neural network using a dedicated tokenizer and padding strategy.

* **Modular Deep Learning Architecture**: The model definition is separated into a dedicated model_builder.py file, making it easy to adjust hyperparameters and layers.

* **Robust Preprocessing**: Includes lowercasing, HTML tag removal, lemmatization, and stop-word removal, all consolidated in one preparation function.

* **Asset Preservation**: Critical artifacts—the trained Keras model, the fitted Keras Tokenizer, and the Label Encoder—are saved to ensure reproducible and portable predictions.


## 📂 Project Structure

```text
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
```

## 🚀 How to Run the Project

### 1. Prerequisites
Make sure you have Python installed.  
It's recommended to create a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Libraries
Install all the required libraries using `pip`
```bash
pip install pandas scikit-learn nltk tensorflow joblib numpy
```

### 3. Download the Dataset
Download the IMDb Dataset of 50k Movie Reviews from Kaggle and place the `IMDB Dataset.csv` file in the project's root directory.

### 4. Run the pipeline
The `train.py` script now handles all steps: preprocessing, model building, training, evaluation, and saving the artifacts.
```bash
python train.py
```
This will output the evaluation results and save the trained model artifacts to the `saved_assets/` directory.

## Evaluation Results

The model achieved the following performance on the test set:

### Accuracy: 86.29%
### Loss : 53.74%



## Make a Live Prediction
You can use the `predict.py` script to test the model with a new review.
```bash
python predict.py
```
The script will demonstrate predictions on example reviews. Feel free to modify the `if __name__ == "__main__": ` block to test your own reviews.
