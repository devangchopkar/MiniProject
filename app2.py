from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load or Train the Model
try:
    model = joblib.load("road_classifier.pkl")
except:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = np.array([[28.7041, 77.1025, 60], [28.5355, 77.3910, 20]])  # Sample data
    y_train = np.array(["Highway", "Service Road"])
    model.fit(X_train, y_train)
    joblib.dump(model, "road_classifier.pkl")

# Get road type from OpenStreetMap (OSM)
def get_road_type(lat, lon):
    try:
        geolocator = Nominatim(user_agent="road_classifier")
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        return location.raw['address'].get('highway', "Unknown") if location else "Unknown"
    except Exception:
        return "Unknown"

@app.route('/classify', methods=['POST'])
def classify_road():
    try:
        data = request.get_json()
        lat = float(data.get("latitude", 0))
        lon = float(data.get("longitude", 0))
        speed = float(data.get("speed", 0))

        if not lat or not lon:
            return jsonify({"error": "Invalid latitude or longitude"}), 400

        # Get road type from OSM and predict using the ML model
        road_type = get_road_type(lat, lon)
        prediction = model.predict([[lat, lon, speed]])[0]

        response = {
            "latitude": lat,
            "longitude": lon,
            "speed": speed,
            "road_type_osm": road_type,
            "predicted_type": prediction
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '_main_':
    # Run on all network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000,debug=True)

