import json
import pickle

import flask
from flask import Blueprint, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from services.cloud_storage import upload_to_storage
from services.cloud_vision import detect_text_uri
from services.gemini_ai import extractingredient
from services.vertex_ai import endpoint_predict_text

predict_bp: Blueprint = Blueprint("predict", __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

with open('tokenizer.pickle', 'rb') as file:
    # Mengambil objek tokenizer dari file menggunakan pickle
    tokenizer = pickle.load(file)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
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
            extracted_entity = extractingredient(ocr_text_new)

            # Tokenize and pad the sequences
            x_new_sequences = tokenizer.texts_to_sequences(ocr_text_new)
            x_new_padded = pad_sequences(x_new_sequences, maxlen=423)

            # Convert extracted_entity to the desired JSON format
            ingredients_json = []
            # predict_list_ingredients = []

            # for entity in extracted_entity["ingredients"]:
            #     x_new_sequences2 = loaded_tokenizer.texts_to_sequences([entity["name"]])
            #     x_new_padded2 = pad_sequences(x_new_sequences2, maxlen=775)
            #     predict_list_ingredients.append(x_new_padded2[0].tolist())

            # Get predictions using Vertex AI predicted_list_ingredients = endpoint_predict_sample2(
            # instances=predict_list_ingredients, project="552288219429")

            formatted_string = ""
            for index, entity in enumerate(extracted_entity["ingredients"]):
                ingredient_dict = {"name": entity["name"]}
                if 'percentage' in entity:
                    ingredient_dict['percentage'] = entity['percentage']
                if 'contains' in entity:
                    ingredient_dict['contains'] = entity['contains']
                # ingredient_dict['status'] = predicted_list_ingredients[index]
                formatted_string += f"{entity['name']}, "
                ingredients_json.append(ingredient_dict)

            # Get additional result using Vertex AI
            result = endpoint_predict_text(instances=x_new_padded.tolist(), project="552288219429")

            return jsonify({
                "success": "Image uploaded and processed successfully",
                "ocr_text": formatted_string,
                "ingredients": ingredients_json,
                "result": result
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid File Type"}), 400

