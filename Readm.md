# Disaster Tweet Classification

This project aims to classify tweets into two categories: Disaster and Non-Disaster. The classification is based on the content of the tweets, where the goal is to predict whether a tweet is related to a real disaster event or not.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Models](#models)
- [Results](#results)
- [Visualization](#visualization)

## Dataset

The dataset used in this project consists of two CSV files:

- **train.csv**: Contains the training data with tweet text and corresponding target labels.
- **test.csv**: Contains the test data with tweet text.

Each row in the dataset contains:

- **text**: The tweet text.
- **target**: The label, where:
  - `1` indicates a disaster tweet.
  - `0` indicates a non-disaster tweet.

## Preprocessing

Before training the models, several preprocessing steps are applied to the data:

1. **Text Cleaning**:
   - Removing URLs, mentions, hashtags, and non-alphabetical characters.
   - Converting the text to lowercase.
   
2. **Text Tokenization**:
   - Splitting the text into words (tokens) using word tokenization.

3. **Stemming & Lemmatization**:
   - Reducing words to their root form using the `PorterStemmer` and `WordNetLemmatizer`.

4. **Stopword Removal**:
   - Filtering out common words like "the", "is", "and", etc., which do not add significant meaning to the text.

## Feature Extraction

We use the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer to extract features from the text data:

- **Unigrams, Bigrams, and Trigrams**: Capture single words, pairs of words, and triples of words to provide richer context.
- **Max Features**: Limit the vocabulary size to 5000 to reduce dimensionality.

## Handling Imbalanced Data

The training data may have an imbalance between disaster and non-disaster tweets. To address this:

1. **Resampling**:
   - Upsample the minority class (non-disaster tweets) to match the majority class size.
   - This ensures the model does not bias towards the majority class.

## Models

We evaluate multiple machine learning models for the classification task:

### 1. Logistic Regression
Logistic Regression is a commonly used binary classifier. It is trained using the **TF-IDF** features extracted from the text data.

- **Accuracy**: 80%
- **Precision**: 0.81
- **Recall**: 0.78
- **F1-Score**: 0.79

### 2. Support Vector Classifier (SVC)
The **Support Vector Classifier (SVC)** with a linear kernel is used for text classification. It is particularly effective for high-dimensional spaces such as text data.

- **Accuracy**: 82%
- **Precision**: 0.83
- **Recall**: 0.80
- **F1-Score**: 0.81

### 3. Random Forest Classifier
A **Random Forest classifier** is trained using the resampled data. This ensemble method combines multiple decision trees to improve classification accuracy and reduce overfitting.

- **Accuracy**: 85%
- **Precision**: 0.86
- **Recall**: 0.84
- **F1-Score**: 0.85

## Results

Each model is evaluated on the test data using the following metrics:

- **Accuracy**: The percentage of correct predictions made by the model.
- **Confusion Matrix**: A matrix showing true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Includes precision, recall, and F1-score for each class (disaster and non-disaster).

### Selected Model
The **Random Forest Classifier** performed the best with an accuracy of 85%. So this model was chosen.

## Visualization

To better understand model performance, confusion matrices are plotted for each model using `matplotlib` and `seaborn`:

- **Confusion Matrix for Logistic Regression**
- **Confusion Matrix for Support Vector Classifier**
- **Confusion Matrix for Random Forest**

These visualizations highlight the distribution of true positives, false positives, true negatives, and false negatives.
