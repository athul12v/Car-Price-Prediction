import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load serialized pipeline artifact once on startup
MODEL_PATH = "./Models/car_price_pipeline.pkl" if os.path.exists("./Models/car_price_pipeline.pkl") else "car_price_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
CURRENT_YEAR = 2026

def build_feature_dataframe(data: dict) -> pd.DataFrame:
    """
    Transforms request payload into the exact DataFrame schema
    expected by the Scikit-Learn ColumnTransformer.
    """
    year = int(data["Year"])
    car_age = CURRENT_YEAR - year

    return pd.DataFrame([{
        "Present_Price": float(data["Present_Price"]),
        "Kms_Driven": float(data["Kms_Driven"]),
        "Car_Age": car_age,
        "Owner": int(data.get("Owner", 0)),
        "Fuel_Type": str(data["Fuel_Type"]),
        "Seller_Type": str(data["Seller_Type"]),
        "Transmission": str(data["Transmission"])
    }])

# UI Route: Web Browser Form
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        try:
            input_df = build_feature_dataframe(request.form)
            predicted_price = model.predict(input_df)[0]
            prediction_text = f"Estimated Resale Value: ₹{predicted_price:.2f} Lakhs"
        except Exception as err:
            prediction_text = f"Prediction Error: {str(err)}"

    return render_template("index.html", prediction_text=prediction_text)

# API Route: Headless REST Endpoint
@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    try:
        payload = request.get_json(force=True)
        input_df = build_feature_dataframe(payload)
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "status": "success",
            "prediction_lakhs": round(float(prediction), 2)
        }), 200
    except Exception as err:
        return jsonify({
            "status": "error",
            "message": str(err)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
