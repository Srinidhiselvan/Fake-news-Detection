import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sn

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def clean_text(text):
    """Cleans the input text by removing punctuation, stopwords, and lemmatizing words."""
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # Remove punctuation and lowercase
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load Dataset
def load_data(file_path):
    """Loads the dataset and preprocesses the text column."""
    try:
        # Try to load the dataset with default settings
        df = pd.read_csv(file_path, on_bad_lines='skip')  # Skip problematic rows
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")
    
    print("Columns in the dataset:", df.columns)  # Debugging step to see column names
    
    # Adjust column names and check for required columns
    if 'Text' in df.columns:
        text_column = 'Text'
    elif 'text' in df.columns:
        text_column = 'text'
    else:
        raise KeyError("Dataset does not have a 'Text' or 'text' column for article content.")
    
    if 'label' not in df.columns:
        raise KeyError("Dataset does not have a 'label' column for target values.")
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=[text_column, 'label'])
    
    # Clean text data
    df['clean_text'] = df[text_column].apply(clean_text)
    print("Sample data after cleaning:\n", df[['clean_text', 'label']].head())
    return df

# Feature Extraction
def extract_features(df):
    """Extracts TF-IDF features from the cleaned text."""
    # TF-IDF vectorization for cleaned text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(df['clean_text']).toarray()

    # Use the 'label' column as the target
    y = df['label'].map({'Fake': 1, 'Real': 0})  # Convert 'Fake' to 1, 'Real' to 0

    # Return features (X), target (y), and vectorizer
    return X_text, y, vectorizer

# Handle Imbalanced Dataset
def balance_dataset(X, y):
    """Balances the dataset using SMOTE."""
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("Class distribution after balancing:", pd.Series(y_balanced).value_counts())
    return X_balanced, y_balanced

# Train and Evaluate Model
def train_and_evaluate(X, y):
    """Trains the model and evaluates its performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression Model (Scikit-learn)
    lr_model = LogisticRegression(max_iter=200)  # Increase max_iter for convergence
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nLogistic Regression Model Performance Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label='Logistic Regression ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Train a Neural Network Model (TensorFlow/Keras)
    nn_model = Sequential([
        Dense(512, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Neural Network Predictions
    nn_y_proba = nn_model.predict(X_test)
    nn_y_pred = (nn_y_proba > 0.5).astype("int32").flatten()

    # Metrics for Neural Network
    nn_acc = accuracy_score(y_test, nn_y_pred)
    nn_precision, nn_recall, nn_f1, _ = precision_recall_fscore_support(y_test, nn_y_pred, average='binary')
    nn_roc_auc = roc_auc_score(y_test, nn_y_proba)

    print("\nNeural Network Model Performance Metrics:")
    print(f"Accuracy: {nn_acc:.4f}")
    print(f"Precision: {nn_precision:.4f}")
    print(f"Recall: {nn_recall:.4f}")
    print(f"F1-Score: {nn_f1:.4f}")
    print(f"ROC-AUC: {nn_roc_auc:.4f}")

    # ROC Curve for Neural Network
    fpr, tpr, _ = roc_curve(y_test, nn_y_proba)
    plt.plot(fpr, tpr, label='Neural Network ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return lr_model, nn_model

# Main Execution
if __name__ == "__main__":
    file_path = r"C:\Users\srini\OneDrive\Documents\fake_and_real_news.csv"  # Replace with your dataset path

    try:
        # Load and process the data
        df = load_data(file_path)

        # Extract features and target
        X, y, vectorizer = extract_features(df)

        # Handle class imbalance
        X_balanced, y_balanced = balance_dataset(X, y)

        # Train and evaluate both models
        train_and_evaluate(X_balanced, y_balanced)

    except Exception as e:
        print(f"Error: {e}")
