# Twitter Sentiment Analysis

This project focuses on building a machine learning model using NLP techniques to analyze the sentiment of tweets — classifying them as either **positive** or **negative**. It uses the **Sentiment140** dataset containing 1.6 million tweets and applies a **Logistic Regression** model for prediction.

---

##  Project Overview

- **Objective**: Automatically classify the sentiment of tweets using text processing and machine learning.
- **Model Used**: Logistic Regression
- **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (from Kaggle)

---

##  Workflow

1. **Data Collection**
   - Downloaded via Kaggle API (using a `kaggle.json` file for authentication)
   
2. **Data Preprocessing**
   - Removal of noise (usernames, URLs, punctuation)
   - Lowercasing
   - Tokenization
   - Stopword removal (`nltk.corpus.stopwords`)
   - Stemming (`nltk.stem.PorterStemmer`)
   
3. **Feature Extraction**
   - Used `TfidfVectorizer` to convert text into numerical feature vectors

4. **Model Training**
   - Used `train_test_split` to divide the data into training and test sets
   - Trained a **Logistic Regression** classifier on the preprocessed text features

5. **Model Evaluation**
   - Accuracy calculation using `accuracy_score`
   - Potential for future extension with precision, recall, F1-score

---

##  Tools and Libraries Used

| Tool/Library | Purpose |
|--------------|---------|
| **Python** | Programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **NLTK** | Natural Language Processing (stopwords, stemming) |
| **re** | Regular expressions for text cleaning |
| **Scikit-learn** | ML model, vectorization, evaluation |
| **Kaggle API** | Dataset download |

---


