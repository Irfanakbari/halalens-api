import time

from flask import Blueprint, request, jsonify

from services.cloud_storage import upload_to_storage
from services.cloud_vision import detect_text_uri

predict_bp: Blueprint = Blueprint("predict", __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()  # Record start time

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
            upload_start_time = time.time()  # Record upload start time
            uri = upload_to_storage(file)
            upload_time = time.time() - upload_start_time  # Calculate upload time

            # Perform OCR on the uploaded image using Google Cloud Vision API
            ocr_start_time = time.time()  # Record OCR start time
            ocr_text = detect_text_uri(uri["uri"])
            ocr_time = time.time() - ocr_start_time  # Calculate OCR time

            return jsonify({
                "ocr_text": ocr_text,

                "image": uri["link"],
                "timings": {
                    "upload_time": upload_time,
                    "ocr_time": ocr_time,
                }
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid File Type"}), 400
