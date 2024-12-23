# Disaster Tweet Classification

This project aims to classify tweets into two categories: Disaster and Non-Disaster. The classification is based on the content of the tweets, where the goal is to predict whether a tweet is related to a real disaster event or not.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)

## Dataset

The dataset used in this project consists of two CSV files:

- **train.csv**: Contains the training data with tweet text and corresponding target labels.
- **test.csv**: Contains the test data with tweet text.

Each row in the dataset contains:

- **text**: The tweet text.
- **target**: The label, where:
  - `1` indicates a disaster tweet.
  - `0` indicates a non-disaster tweet.

### Sample Data (train.csv):

| text                                         | target |
|----------------------------------------------|--------|
| Just saw a car crash! It's terrible.         | 1      |
| Happy birthday to my friend!                | 0      |
| Massive earthquake hits the city.            | 1      |
| The sun is shining, it's a beautiful day!    | 0      |

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

## Models

We evaluate multiple machine learning models for the classification task:

### 1. Logistic Regression
Logistic Regression is a commonly used binary classifier. It is trained using the **TF-IDF** features extracted from the text data. This model is simple and interpretable.

### 2. Support Vector Classifier (SVC)
The **Support Vector Classifier (SVC)** with a linear kernel is used for text classification. It is particularly effective for high-dimensional spaces such as text data.

### 3. Random Forest Classifier
A **Random Forest classifier** is trained using the resampled data. This ensemble method combines multiple decision trees to improve classification accuracy and reduce overfitting.

## Results

Each model is evaluated on the test data using the following metrics:

- **Accuracy**: The percentage of correct predictions made by the model.
- **Confusion Matrix**: A matrix showing true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Includes precision, recall, and F1-score for each class (disaster and non-disaster).

The performance of each model is visualized using confusion matrices, which are plotted using `matplotlib` and `seaborn`.

### Example Model Performance:
- **Accuracy**: 80%
- **Confusion Matrix**:
    ```
    [[500, 100]
     [ 50, 350]]
    ```
- **Classification Report**:
    ```
    Precision: 0.78
    Recall: 0.82
    F1-score: 0.80
    ```

## Usage

To use the trained **Random Forest** model to make predictions on new tweets, use the following approach:

1. Preprocess the text data as described in the preprocessing section.
2. Transform the new text data into TF-IDF features using the same `TfidfVectorizer` used for training.
3. Use the trained model to predict the disaster or non-disaster category for each tweet.

This will output the prediction for each tweet, indicating whether it's related to a real disaster or not.

