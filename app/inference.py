import joblib

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def predict(text):
    vectorized = vectorizer.transform([text])
    proba = model.predict_proba(vectorized)[0]
    confidence = round(max(proba) * 100, 2)

    result = model.predict(vectorized)[0]
    prediction = "Positive 😊" if result == 1 else "Negative 😡"

    return prediction, confidence
