import time

import flask
from flask import Blueprint, request, jsonify

from services.cloud_storage import upload_to_storage
from services.cloud_vision import detect_text_uri
from services.gemini_ai import infosyubhat
from services.vertex_ai import endpoint_predict_text2

predict_bp: Blueprint = Blueprint("predict", __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# with open('tokenizer.pickle', 'rb') as file:
#     # Mengambil objek tokenizer dari file menggunakan pickle
#     tokenizer = pickle.load(file)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()  # Record start time

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
            upload_start_time = time.time()  # Record upload start time
            uri = upload_to_storage(file, uid)
            upload_time = time.time() - upload_start_time  # Calculate upload time

            # Perform OCR on the uploaded image using Google Cloud Vision API
            ocr_start_time = time.time()  # Record OCR start time
            ocr_text = detect_text_uri(uri["uri"])
            ocr_time = time.time() - ocr_start_time  # Calculate OCR time

            # ocr_text_new = [ocr_text]
            # Process the OCR text
            # extraction_start_time = time.time()  # Record extraction start time
            # extracted_entity = extractingredient(ocr_text)
            # extraction_time = time.time() - extraction_start_time  # Calculate extraction time

            # Convert extracted_entity to the desired JSON format
            ingredients_json = []
            # predict_list_ingredients = []

            # for entity in extracted_entity["ingredients"]:
            #     x_new_sequences2 = loaded_tokenizer.texts_to_sequences([entity["name"]])
            #     x_new_padded2 = pad_sequences(x_new_sequences2, maxlen=775)
            #     predict_list_ingredients.append(x_new_padded2[0].tolist())

            # Get predictions using Vertex AI predicted_list_ingredients = endpoint_predict_sample2(
            # instances=predict_list_ingredients, project="552288219429")

            # formatted_string = ""
            # for index, entity in enumerate(extracted_entity["ingredients"]):
            #     ingredient_dict = {"name": entity["name"]}
            #     if 'percentage' in entity:
            #         ingredient_dict['percentage'] = entity['percentage']
            #     if 'contains' in entity:
            #         ingredient_dict['contains'] = entity['contains']
            #     # ingredient_dict['status'] = predicted_list_ingredients[index]
            #     formatted_string += f"{entity['name']}, "
            #     ingredients_json.append(ingredient_dict)

            # Get additional result using Vertex AI
            # Tokenize and pad the sequences
            # x_new_sequences = tokenizer.texts_to_sequences([formatted_string])
            # x_new_padded = pad_sequences(x_new_sequences, maxlen=423)

            # Get additional result using Vertex AI
            result_start_time = time.time()  # Record result prediction start time
            result = endpoint_predict_text2(ocr_text)
            result_time = time.time() - result_start_time  # Calculate result prediction time

            info = ''
            if result == 'haram':
                info_start_time = time.time()  # Record information retrieval start time
                info = infosyubhat(ocr_text)
                info_time = time.time() - info_start_time  # Calculate information retrieval time

            total_time = time.time() - start_time  # Calculate total time
            return jsonify({
                "ocr_text": ocr_text,
                # "ingredients": ingredients_json,
                "info": info,
                "result": result,
                "image": uri["link"],
                "timings": {
                    "upload_time": upload_time,
                    "ocr_time": ocr_time,
                    # "extraction_time": extraction_time,
                    "result_time": result_time,
                    "info_time": info_time if result == 'haram' else None,
                    "total_time": total_time
                }
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid File Type"}), 400
