import os

import numpy as np
import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel


def endpoint_predict_text(instances, project=os.environ.get('PROJECT'), location=os.environ.get('LOCATION'),
                          endpoint_id=os.environ.get('LSTM_ENDPOINT')):
    aiplatform.init(project=project)
    endpoint = aiplatform.Endpoint(f"projects/{project}/locations/{location}/endpoints/{endpoint_id}")
    prediction = endpoint.predict(instances=instances)

    # Extract output probabilities
    output_probabilities = prediction[0]
    label_mapping = {0: 'halal', 1: 'haram'}

    # Get predicted labels
    predicted_labels = [np.argmax(probabilities) for probabilities in output_probabilities]
    predicted_categories = [label_mapping[label] for label in predicted_labels]

    # Return the predicted category
    return predicted_categories[0]


def endpoint_predict_text2(text, project=os.environ.get('PROJECT_ID'), location=os.environ.get('LOCATION'),
                           endpoint_id=os.environ.get('BISON_ENDPOINT')):
    vertexai.init(project=project, location=location)
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 512,
        "temperature": 0.9,
        "top_p": 1
    }
    model = TextGenerationModel.from_pretrained("text-bison@002")
    model = model.get_tuned_model(f"projects/{project}/locations/{location}/models/{endpoint_id}")
    response = model.predict(
        f"""Classify the following ingredient into one of the following classes: [halal, haram] Text: {text}. (if 
        from animal [haram], from plant [halal], unknown source [haram])""",
        **parameters
    )
    # Return the list of predicted categories
    return response.text


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
        f"""Classify the following ingredient into one of the following classes: [halal, haram, syubhat] Text:{keyword}
        """
        , **parameters
    )
    return response.text


def about_ingredient(keyword):
    model = GenerativeModel("gemini-pro")
    responses = model.generate_content(
        f"""jelaskan 1 kalimat tentang pengetian atau fungsi bahan makanan dalam produk makanan : {keyword}""",
        generation_config={
            "max_output_tokens": 300,
            "temperature": 0.9,
            "top_p": 1
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
                generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED:
                generative_models.HarmBlockThreshold.BLOCK_NONE
        },
        stream=False,
    )
    return responses.text
