# sentiment-analysis-app

 Sentiment Analysis Web App (Flask + ML)
 Overview

This project is a Machine Learning powered web application that classifies movie reviews as Positive or Negative using Natural Language Processing (NLP).

Built using:
Python
Scikit-learn
TF-IDF Vectorization
Logistic Regression
Flask

Problem Statement

Manual analysis of large volumes of text reviews is inefficient.
This project automates sentiment classification using a trained ML model on the IMDB dataset.

Solution Approach

Preprocessed IMDB dataset (50K reviews)
Converted text to numerical features using TF-IDF
Trained Logistic Regression classifier
Saved trained model
Built Flask web interface for real-time predictions

Project Structure
Sentiment-Analysis-App/
│
├── app.py
├── train.py
├── model/
├── templates/
├── requirements.txt
└── README.md
Installation
git clone https://github.com/padmajakotoky73-hash/sentiment-analysis-app.git
cd sentiment-analysis-app
pip install -r requirements.txt
python app.py
Demo
Runs locally at:
http://127.0.0.1:5000
Dataset:
IMDB Movie Reviews Dataset (50,000 samples)
Not included in repository due to size constraints.

Future Improvements
Deploy to cloud (Render / Railway)
Add confidence score
Improve UI styling
Try deep learning (LSTM / BERT)
