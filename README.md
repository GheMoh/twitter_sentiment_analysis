# Twitter Sentiment Analysis

## Project Overview

The **Twitter Sentiment Analysis** project is an end-to-end data analysis and machine learning project designed to perform sentiment analysis on Twitter data. It leverages various Python libraries and natural language processing techniques to determine whether tweets express positive, negative, or neutral sentiments. The project is organized as a Jupyter Notebook, enabling interactive code execution and data visualization.

## Project Components

### 1. Data Acquisition and Preprocessing
   - Import essential libraries, including Pandas, Matplotlib, Scikit-learn, Keras, NLTK, Gensim, and more.
   - Load the Twitter dataset containing columns like target, ids, date, flag, user, and text.
   - Map target labels (0 = negative, 2 = neutral, 4 = positive) to string labels (NEGATIVE, NEUTRAL, POSITIVE).
   - Preprocess text data by removing special characters, links, stopwords, and optional stemming.

### 2. Data Exploration
   - Visualize the distribution of sentiment labels in the dataset using bar charts.

### 3. Word Embedding with Word2Vec
   - Train a Word2Vec model on the text data to convert words into dense vector representations.
   - Word embeddings capture semantic relationships between words.

### 4. Text Tokenization
   - Tokenize text data to convert words into numerical sequences suitable for machine learning.
   - Use a Tokenizer to create a vocabulary of unique words, and pad text sequences to a fixed length.

### 5. Label Encoding
   - Encode sentiment labels into numerical values for machine learning.
   - Utilize a LabelEncoder to map sentiment categories (e.g., NEGATIVE, NEUTRAL, POSITIVE) to numerical values.

### 6. Model Building with Keras
   - Construct a neural network model using the Keras library.
   - The model comprises an embedding layer, LSTM (Long Short-Term Memory) layer, dropout layers, and a dense layer with sigmoid activation for binary classification.
   - Display the model summary and architecture.

### 7. Model Compilation and Training
   - Compile the model with binary cross-entropy loss and the Adam optimizer.
   - Train the model on the dataset, incorporating callbacks like learning rate reduction and early stopping for improved efficiency.

### 8. Model Evaluation
   - Evaluate the trained model on the test dataset.
   - Report accuracy, loss, and other relevant metrics.
   - Visualize learning curves to assess training and validation performance.

### 9. Sentiment Prediction
   - Define a function for predicting sentiment labels (NEGATIVE, NEUTRAL, POSITIVE) for input text.
   - The function returns the predicted label, sentiment score, and execution time.

### 10. Confusion Matrix and Classification Report
    - Generate a confusion matrix to evaluate the model's performance.
    - Produce a detailed classification report with metrics for each sentiment category.

### 11. Model Saving
    - Save the trained Keras model, Word2Vec model, Tokenizer, and LabelEncoder for future use.

## Project Purpose

The main purpose of this project is to illustrate the sentiment analysis process for Twitter data using machine learning techniques. It allows users to analyze tweet sentiments and gain insights into how different machine learning components can be combined to perform sentiment classification. This project serves as a valuable starting point for sentiment analysis tasks in social media monitoring, brand analysis, or customer feedback analysis. Additionally, it showcases best practices for text data preprocessing and building neural network models for natural language processing tasks.
