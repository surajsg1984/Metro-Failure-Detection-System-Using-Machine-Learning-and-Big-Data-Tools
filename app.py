from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load LSTM model from pickle
with open(r"C:\Users\Admin\Downloads\Project\model.pkl", "rb") as f:
    model_json, weights = pickle.load(f)

model = model_from_json(model_json)
model.set_weights(weights)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Example input shape (replace with your actual values)
seq_len = 10
num_features = 5

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fetch input values from HTML form
        features = [float(x) for x in request.form.values()]
        
        # Convert into numpy array (reshape as per model)
        input_array = np.array(features).reshape(1, seq_len, num_features)
        
        # Run prediction
        prediction = model.predict(input_array)
        result = "Failure Detected" if prediction[0][0] > 0.5 else "Normal Operation"

        return render_template("index.html", prediction_text="Prediction: " + result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
