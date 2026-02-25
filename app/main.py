from flask import Flask, render_template, request
from app.preprocessing import clean_text
from app.inference import predict

app = Flask(__name__, template_folder="../templates")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        user_input = request.form["review"]
        cleaned = clean_text(user_input)
        prediction, confidence = predict(cleaned)

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
