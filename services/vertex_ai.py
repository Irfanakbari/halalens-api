import numpy as np
from google.cloud import aiplatform


def endpoint_predict_sample(
        instances,
        project: str = "552288219429",
        location: str = "us-central1",
        endpoint_id: str = "1729709911375347712"
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(f"projects/{project}/locations/{location}/endpoints/{endpoint_id}")

    prediction = endpoint.predict(instances=instances)
    # Extract output probabilities
    output_probabilities = prediction[0]

    label_mapping = {0: 'halal', 1: 'haram', 2: 'syubhat'}

    # Get predicted labels
    predicted_labels = [np.argmax(probabilities) for probabilities in output_probabilities]
    predicted_categories = [label_mapping[label] for label in predicted_labels]

    # Print the results
    for category in predicted_categories:
        return category
