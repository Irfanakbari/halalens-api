import json

from vertexai.language_models import TextGenerationModel
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
        "max_output_tokens": 138,
        "temperature": 0.9,
        "top_p": 1
    }
    model = TextGenerationModel.from_pretrained("gemini-pro")
    response = model.predict(
        f"""jelaskan secara singkat 1 kalimat tentang bahan baku makanan yang saya sebutkan, kenapa bisa syubhat, terutama pada bahan hewani

    input: describe this ingredients why syubhat, Text: Corn grits (domestic production), sugar, vegetable oil, margarine, sweetened condensed milk, dextrin, strawberry puree, salt, glucose, powdered vinegar/sorbitol, flavoring, acidulant, moss color, emulsifier, sweetener (sucralose), calcium carbonate, gardenia pigment, carotenoid pigment,
    (Contains milk ingredients and soybeans in part)
    output: Margarin bisa mengandung bahan non halal

    input: describe this ingredients why syubhat, Text: sweetened condensed milk, dextrin, strawberry puree, salt, enzyme, powdered vinegar/sorbitol, flavoring, acidulant, moss color, emulsifier, sweetener (sucralose), calcium carbonate, gardenia pigment, carotenoid pigment,
    output: Enzyme mungkin bisa berbahan hewani non halal atau nabati

    input: describe this ingredients why syubhat, Text: enzyme, gelatin
    output: Enzyme bisa berbahan hewani hewani non halal atau nabati
    Gelatin bisa berasal dari hewan non halal

    input: describe this ingredients why syubhat, Text:milk chocolate sugar chocolate liquor cocoa butter milk soya lecithin emulsifier pure vanilla ginger green tea,
    output: Emulsifier bisa berbahan nabati atau hewani non halal

    input: describe this ingredients why syubhat, Text: {ingredients}
    output:
    """,
        **parameters
    )
    return response.text
