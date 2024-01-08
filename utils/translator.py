from deep_translator import GoogleTranslator


def translate_to_english(text, target_language='en'):
    translator = GoogleTranslator(source='auto', target=target_language)
    translation = translator.translate(text)
    return translation
