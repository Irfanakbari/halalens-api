import json
import flask
from flask import Blueprint, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from services.cloud_storage import upload_to_storage
from services.cloud_vision import detect_text_uri
from services.vertex_ai import endpoint_predict_sample

predict_bp: Blueprint = Blueprint("predict", __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

with open('tokenizer.json', 'r', encoding='utf-8') as json_file:
    loaded_tokenizer_json = json_file.read()
    loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@predict_bp.route('/predict', methods=['POST'])
def predict():
    decoded_token = flask.g.get('decoded_token')
    if not decoded_token:
        return jsonify({"error": "Invalid Access Token"}), 401

    # Extract user UID from the decoded token
    uid = decoded_token['uid']

    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['file']

    # If the user submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No Selected File"}), 400

    # If the file is provided and is an allowed file type (e.g., image)
    if file and allowed_file(file.filename):
        # Upload the file to Google Cloud Storage
        uri = upload_to_storage(file, uid)

        # Perform OCR on the uploaded image using Google Cloud Vision API
        ocr_text = detect_text_uri(uri)
        ocr_text_new = [ocr_text]
        X_new_sequences = loaded_tokenizer.texts_to_sequences(ocr_text_new)
        X_new_padded = pad_sequences(X_new_sequences, maxlen=775)

        # Perform your image processing or prediction logic here
        # ...

        result = endpoint_predict_sample(project="552288219429", instances=X_new_padded.tolist())

        return jsonify({
            "success": "Image uploaded and processed successfully",
            "ocr_text": ocr_text,
            "result": result
        }), 200

    return jsonify({"error": "Invalid File Type"}), 400
