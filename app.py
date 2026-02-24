from flask import Flask, render_template, request
import joblib
import re
import string
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        user_input = request.form["review"]
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        prediction = "Positive 😊" if result == 1 else "Negative 😡"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
