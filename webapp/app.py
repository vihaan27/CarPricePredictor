from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load("model.joblib")

# My App
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        ## Testing model
        sample = np.array([[12,73000,20.36,1197,78.9,5,2982,0,0,0,0,0,1,1]])
        prediction = model.predict(sample)
        return render_template("index.html", prediction_text=f"Predicted Price: ₹{int(prediction):,}") # Placeholder

if __name__ == "__main__":
    app.run(debug=True)