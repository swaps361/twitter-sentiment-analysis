# Sentiment Analysis and Entity Recognition on Twitter Data

This project demonstrates sentiment analysis and entity recognition on Twitter data using Natural Language Processing (NLP) techniques. The main goal is to preprocess tweets, extract features, and classify the sentiment of tweets as either "Positive" or "Negative". Additionally, the project includes named entity recognition (NER) to identify entities mentioned in the tweets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Entity Recognition](#entity-recognition)
- [Classification Models](#classification-models)
- [Results](#results)


## Introduction

The project uses the `twitter_samples` dataset from the NLTK library, which contains a collection of positive and negative tweets. The data is preprocessed by tokenizing, removing stopwords, and applying stemming and lemmatization. Features are then extracted using a combination of unigrams and bigrams. Finally, a Naive Bayes classifier and a Decision Tree classifier are trained to classify the sentiment of the tweets.

In addition to sentiment analysis, the project also includes an entity recognition component, which identifies named entities within the tweets using spaCy's pre-trained language model.

## Features

- Sentiment analysis of Twitter data
- Preprocessing of tweets (tokenization, stopword removal, stemming, lemmatization)
- Feature extraction using unigrams and bigrams
- Sentiment classification using Naive Bayes and Decision Tree classifiers
- Named Entity Recognition (NER) using spaCy
- Cross-validation for model evaluation

## Dependencies

The project requires the following Python libraries:

- `nltk`
- `spacy`
- `scikit-learn`
- `numpy`


## Data Preprocessing
The tweets are preprocessed by performing the following steps:

Tokenization: Breaking the text into individual words.
Stopword Removal: Removing common stopwords that do not contribute to the sentiment.
Stemming and Lemmatization: Normalizing the words by reducing them to their root forms.
## Feature Extraction
The features are extracted by combining unigrams and bigrams from the preprocessed tweets. The most frequent 2000 words are selected as features for the classification models.

## Entity Recognition
The entity recognition component uses spaCy to identify named entities in the tweets, such as people, organizations, and locations. This is useful for understanding the context in which entities are mentioned within the tweets.

## Classification Models
Two classification models are trained on the extracted features:

Naive Bayes Classifier: A simple yet effective probabilistic classifier.
Decision Tree Classifier: A tree-based model that splits the data based on feature importance.
Both models are evaluated on a test set, and cross-validation is performed to assess model robustness.

## Results
Naive Bayes Accuracy: ~75% (example accuracy, varies based on actual data)
Decision Tree Accuracy: ~65% (example accuracy, varies based on actual data)
Cross-validation accuracy: ~70% (example accuracy, varies based on actual data)
These results demonstrate the effectiveness of the models in classifying the sentiment of tweets.
