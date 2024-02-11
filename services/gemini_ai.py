import json

from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel


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


def infosyubhat(ingredients):
    parameters = {
        "max_output_tokens": 150,
        "temperature": 0.9,
        "top_p": 1
    }
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(
        f"""jelaskan secara singkat 1 kalimat tentang bahan baku makanan yang saya sebutkan, kenapa bisa haram, terutama pada bahan hewani

    input: describe this ingredients why syubhat, Text: Corn grits (domestic production), sugar, vegetable oil, margarine, sweetened condensed milk, dextrin, strawberry puree, salt, glucose, powdered vinegar/sorbitol, flavoring, acidulant, moss color, emulsifier, sweetener (sucralose), calcium carbonate, gardenia pigment, carotenoid pigment,
    (Contains milk ingredients and soybeans in part)
    output: Margarin dapat mengandung bahan non halal

    input: describe this ingredients why syubhat, Text: sweetened condensed milk, dextrin, strawberry puree, salt, enzyme, powdered vinegar/sorbitol, flavoring, acidulant, moss color, emulsifier, sweetener (sucralose), calcium carbonate, gardenia pigment, carotenoid pigment,
    output: Enzyme dapat berbahan hewani non halal

    input: describe this ingredients why syubhat, Text: enzyme, gelatin
    output: Enzyme dapat berbahan hewani non halal
    Gelatin dapat dari hewan non halal

    input: describe this ingredients why syubhat, Text:milk chocolate sugar chocolate liquor cocoa butter milk soya lecithin emulsifier pure vanilla ginger green tea,
    output: Emulsifier dapat berbahan hewani non halal

    input: describe this ingredients why syubhat, Text: {ingredients}
    output:
    """,
        generation_config=parameters,
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
                generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED:
                generative_models.HarmBlockThreshold.BLOCK_NONE
        },)
    return response.text
