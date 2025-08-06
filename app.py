from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("lstm_model.keras")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get numeric features
        num_features = [
            float(request.form["TP2"]),
            float(request.form["TP3"]),
            float(request.form["H1"]),
            float(request.form["DV_pressure"]),
            float(request.form["Reservoirs"]),
            float(request.form["Oil_temperature"]),
            float(request.form["Motor_current"]),
            float(request.form["COMP"])
        ]

        # Get dropdown features (as integers)
        dropdown_features = [
            int(request.form["DV_eletric"]),
            int(request.form["Towers"]),
            int(request.form["MPG"]),
            int(request.form["LPS"]),
            int(request.form["Pressure_switch"]),
            int(request.form["Oil_level"]),
            int(request.form["Caudal_impulses"])
        ]

        # Combine all features
        all_features = num_features + dropdown_features

        # Reshape for LSTM: (samples, timesteps, features)
        input_data = np.array(all_features).reshape((1, 1, len(all_features)))

        # Predict
        prediction = model.predict(input_data)
        prediction_value = float(prediction[0][0])

        return render_template("index.html", prediction=prediction_value)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
