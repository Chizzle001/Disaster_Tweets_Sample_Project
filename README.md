# Disaster Tweet Classification

This project is focused on classifying tweets into two categories: Disaster and Non-Disaster. The goal is to predict whether a tweet is related to a real disaster event based on its content.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project uses natural language processing (NLP) techniques to classify tweets. The classification process involves the following key steps:

1. **Text Cleaning**: Removing URLs, mentions, hashtags, and special characters.
2. **Text Preprocessing**: Tokenizing, stemming, and lemmatizing the text.
3. **Feature Extraction**: Using TF-IDF and N-grams (unigrams, bigrams, trigrams) for feature extraction.
4. **Class Imbalance Handling**: Resampling the minority class using upsampling.
5. **Modeling**: Training models such as Logistic Regression, Support Vector Classifier (SVC), and Random Forest Classifier.
6. **Model Evaluation**: Evaluating the models using accuracy, confusion matrix, and classification report.
7. **Prediction**: Predicting whether new tweets are related to a disaster or not.

## Installation

To set up this project, you need Python and several libraries. Follow the instructions below to install the required dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/disaster-tweet-classification.git
   cd disaster-tweet-classification
