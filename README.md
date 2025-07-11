# Spam Mail Detection

This project is a simple machine learning model that can detect whether a message is **spam** or **not spam (ham)**. It uses text processing and machine learning to make predictions.

## What It Does

- Cleans and processes the text data
- Converts text into numbers using TF-IDF
- Trains machine learning models like Naive Bayes and SVM
- Evaluates the model with accuracy and precision
- Saves the model so you can use it later


## How to Run


1. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud

python spam_detector.py
