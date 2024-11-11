from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision.models import swin_b
import joblib

app = Flask(__name__)

# Locate appropriate hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load House Swin Transformer
model = swin_b(num_classes=1, weights=None)
model.load_state_dict(torch.load("./price_model.pth"))

# Load baseline XGBoost model
xgboost_model = joblib.load("./xgboost_model.joblib")
xgboost_model.device = device

# Is Alive?
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Model server is running"}), 200

# House Swin Transformer
@app.route('/image_predict', methods=['POST'])
def image_predict():
    # Find image in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    print(request.files['image'])
    # Get image data
    image_file = request.files['image'].read()

    try:
        # Decode image from POST call
        image_np = np.frombuffer(image_file, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Resize image to correct input and normalize
        image = torch.Tensor(cv2.resize(image, (224, 224))) / 255.
        # Reshape tensor to correct input
        image = image.unsqueeze(0).permute(0, 3, 1, 2)

        # Predict price from model
        prediction = model(image)

        # Convert to response to list
        response = {"prediction": prediction.tolist()}

    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response), 200

@app.route('/xgboost_predict', methods=['POST'])
def xgboost_predict():
    # Receive request data
    data = request.get_json()

    # Convert to XGBoost readable data
    numeric_input = np.expand_dims(np.array(data["input"]), axis=0)

    try:
        # Predict price on input
        prediction = xgboost_model.predict(numeric_input)
        # Convert response to list
        response = {"prediction": prediction.tolist()}
    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response), 200

# Run on public IP from port 4000
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4000)
