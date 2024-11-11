from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision.models import swin_b
import joblib

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = swin_b(num_classes=1, weights=None)
model.load_state_dict(torch.load("./price_model.pth"))

xgboost_model = joblib.load("./xgboost_model.joblib")
xgboost_model.device = device

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Model server is running"}), 200

@app.route('/image_predict', methods=['POST'])
def image_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image'].read()
    
    try:
        image_np = np.frombuffer(image_file, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        image = torch.Tensor(cv2.resize(image, (224, 224))) / 255.
        image = image.unsqueeze(0).permute(0, 3, 1, 2)

        prediction = model(image)

        response = {"prediction": prediction.tolist()}

    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response), 200

@app.route('/xgboost_predict', methods=['POST'])
def xgboost_predict():
    data = request.get_json()

    numeric_input = np.expand_dims(np.array(data["input"]), axis=0)

    try:
        prediction = xgboost_model.predict(numeric_input)
        response = {"prediction": prediction.tolist()}
    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4000)
