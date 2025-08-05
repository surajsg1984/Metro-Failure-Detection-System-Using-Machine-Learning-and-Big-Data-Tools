from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the LSTM model
import requests
url = "https://github.com/surajsg1984/Metro-Failure-Detection-System-Using-Machine-Learning-and-Big-Data-Tools/blob/main/lstm_model.keras"
open("lstm_model.keras", "wb").write(requests.get(url).content)
model = load_model("lstm_model.keras")


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get features from the form
        features = [
            float(request.form["f1"]),
            float(request.form["f2"]),
            float(request.form["f3"]),
            float(request.form["f4"])
        ]
        
        # Convert to numpy array and reshape for LSTM
        # Assuming LSTM expects shape: (samples, timesteps, features)
        input_data = np.array(features).reshape((1, 1, len(features)))
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return render_template("index.html", prediction=str(prediction[0][0]))
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

