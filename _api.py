from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the pre-trained model
try:
    model = joblib.load("xgboost_model.joblib")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        living_area = data.get('living_area')
        bathrooms = data.get('bathrooms')
        bedrooms = data.get('bedrooms')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        # Ensure all values are present
        if None in [living_area, bathrooms, bedrooms, latitude, longitude]:
            return jsonify({"error": "Missing required data fields"}), 400

        # Prepare data for the model
        input_data = pd.DataFrame({
            "Latitude": [latitude],
            "Longitude": [longitude],
            "Living Area": [living_area],
            "Bathrooms": [bathrooms],
            "Bedrooms": [bedrooms]
        })

        # Make a prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction
        return jsonify({'predicted_price': prediction})

    except Exception as e:
        # Print the error and return a JSON response with the error message
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
