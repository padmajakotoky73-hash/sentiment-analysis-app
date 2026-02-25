# Sentiment Analysis Web App

## Problem Statement
Build a machine learning model that classifies movie reviews as positive or negative.

## Dataset
IMDb Reviews Dataset (50,000 reviews)

## Solution Approach
- Text cleaning (lowercase, remove HTML, punctuation)
- Stopword removal (NLTK)
- TF-IDF Vectorization
- Logistic Regression classifier
- Model saved using joblib
- Flask web app for deployment

## Tech Stack
- Python
- Flask
- Scikit-learn
- NLTK
- Pandas

## How to Run Locally

1. Create virtual environment: py -3.11 -m venv .venv
.venv\Scripts\activate

2. Install dependencies:pip install -r requirements.txt

3. Run:python app.py

Open:
http://127.0.0.1:5000

## Future Improvements
- Add neutral sentiment
- Deploy to cloud
- Improve UI
- Add probability scores

---
Built by Padmaja 