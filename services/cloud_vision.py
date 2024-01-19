from google.cloud import vision

from utils.clean_text import clean_text
from utils.translator import translate_to_english


def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    # Specify the feature to enable text detection
    # features = [{"type": vision.Feature.Type.TEXT_DETECTION}]

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        clean = clean_text(texts[0].description.lower())
        translate = translate_to_english(clean)
        return translate  # Return the first text annotation in lowercase

    return "No text detected"
