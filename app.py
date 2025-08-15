from flask import Flask, request, jsonify, send_from_directory
import joblib
import traceback
import numpy as np
from xgboost import XGBClassifier   

app = Flask(__name__, static_folder='static')

# Load your trained model
model = joblib.load("jn_model.pkl")  # Ensure the file exists

EXPECTED_FEATURE_COUNT = 13

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "Missing or invalid JSON"}), 400

        features = payload.get("features")
        if not isinstance(features, list):
            return jsonify({"error": '"features" must be a list'}), 400

        if len(features) != EXPECTED_FEATURE_COUNT:
            return jsonify({"error": f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(features)}"}), 400

        # Convert to NumPy array with shape (1, 13)
        features_array = np.array(features).reshape(1, -1)
        print("Features array shape:", features_array.shape)
        print(features_array)

        # Get probability for CHD positive
        prob = model.predict_proba(features_array)[0][1]
        risk_score = float(prob * 100)
        print("Risk score:", risk_score)

        # Risk level
        if risk_score < 30:
            risk_level = "Low risk"
        elif risk_score < 70:
            risk_level = "Moderate risk"
        else:
            risk_level = "High risk"

        # Return safe Python float for JSON
        return jsonify({
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level
        }), 200

    except Exception as e:
        app.logger.error("Prediction error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory("static", "index2.html")

@app.route("/predictor", methods=["GET"])
def serve_predictor():
    return send_from_directory("static", "predictor.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
