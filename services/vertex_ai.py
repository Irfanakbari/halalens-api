import os

import numpy as np
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

from utils.translator import translate_to_english


def endpoint_predict_text(instances, project=os.environ.get('PROJECT'), location=os.environ.get('LOCATION'),
                          endpoint_id=os.environ.get('LSTM_ENDPOINT')):
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(f"projects/{project}/locations/{location}/endpoints/{endpoint_id}")
    prediction = endpoint.predict(instances=instances)

    # Extract output probabilities
    output_probabilities = prediction[0]
    label_mapping = {0: 'halal', 1: 'haram', 2: 'syubhat'}

    # Get predicted labels
    predicted_labels = [np.argmax(probabilities) for probabilities in output_probabilities]
    predicted_categories = [label_mapping[label] for label in predicted_labels]

    # Return the predicted category
    return predicted_categories[0]


def endpoint_predict_sample2(instances, project=os.environ.get('PROJECT'), location=os.environ.get('LOCATION'),
                             endpoint_id=os.environ.get('LSTM_ENDPOINT')):
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(f"projects/{project}/locations/{location}/endpoints/{endpoint_id}")
    predictions = endpoint.predict(instances=instances)

    label_mapping = {0: 'halal', 1: 'haram', 2: 'syubhat'}
    # Extract output probabilities
    predicted_categories = []

    for prediction in predictions[0]:
        # Get the index of the category with the highest probability
        predicted_label_index = np.argmax(prediction)
        # Convert the category index to the corresponding label according to the mapping
        predicted_category = label_mapping[predicted_label_index]
        # Add the predicted label to the list predicted_categories
        predicted_categories.append(predicted_category)

    # Return the list of predicted categories
    return predicted_categories


def search_ingredient(keyword):
    aiplatform.init(project=os.environ.get('PROJECT'), location=os.environ.get('LOCATION'))
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1
    }
    model = TextGenerationModel.from_pretrained("text-bison@002")
    model = model.get_tuned_model(f"projects/{os.environ.get('PROJECT')}/locations/{os.environ.get('LOCATION')}/models/"
                                  f"{os.environ.get('BISON_ENDPOINT')}")
    response = model.predict(
        f"""Classify the following ingredient into one of the following classes: [halal, haram, syubhat] Text:{translate}"""
        , **parameters
    )
    return response.text


def about_ingredient(keyword):
    model = GenerativeModel("gemini-pro")
    responses = model.generate_content(
        f"""jelaskan 1 kalimat tentang pengetian atau fungsi bahan makanan dalam produk makanan : {keyword}""",
        generation_config={
            "max_output_tokens": 200,
            "temperature": 0.8,
            "top_p": 1
        },
        stream=False,
    )
    return responses.text
