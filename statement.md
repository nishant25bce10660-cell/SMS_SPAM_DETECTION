1. Problem Statement

SMS spam has become a widespread issue, causing inconvenience, privacy breaches, fraud, and security threats.
Users often receive unwanted promotional content, phishing attempts, fake lottery scams, and abusive messages.
Manual filtering is not possible at scale, so an automated system is required.

This project aims to build a Machine Learning–based SMS Spam Detection System that automatically classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.


2. Scope of the Project

The scope of this system includes:

Collecting and preprocessing SMS messages

Converting text into TF-IDF numerical features

Training a Machine Learning model for classification

Providing prediction output (Spam/Ham)

Showing probability score for better confidence

Supporting both command-line and interactive input

Implementing EDA (Exploratory Data Analysis) for insights

Saving trained model for future predictions

Out of scope (for this version):

No mobile app or GUI (future enhancement)

No real-time SMS integration

No cloud deployment


3. Target Users

This system is useful for:

  Individual users

Those who want to filter spam messages on their devices

Students learning Machine Learning & NLP

 Developers

Anyone building SMS filtering systems

ML learners who want to understand text classification

 Organizations

Businesses receiving bulk SMS and needing spam filtering

Teams working on cybersecurity and automation


4. High-Level Features

Spam/Ham classification using ML

Text preprocessing (cleaning, lowercasing, noise removal)

TF-IDF vectorization to convert text into features

High accuracy model (96–98%)

Interactive and CLI prediction modes

Spam probability output

Rule-based boosting for slang/profanity

EDA scripts for data insights

Modular code structure (train.py, predict.py, models/, data/)

