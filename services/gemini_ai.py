import json
import re

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part


def extractingredient(ingredient):
    model = GenerativeModel("gemini-pro")
    responses = model.generate_content(
        f"""extract each ingredient entity name with format  'ingredients': 'name' 
        percentage, contains, and convert to json format and fix if typo : $ingredient = {ingredient}""",
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1
        },
        stream=False,
    )
    finalresp = []
    newjson = str(responses.text.replace('json', '')).strip("`")
    jsonfinal = json.loads(newjson)
    finalresp.append(jsonfinal)

    return finalresp[0]  # Corrected the syntax here
