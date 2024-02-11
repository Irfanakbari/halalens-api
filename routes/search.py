from flask import Blueprint, jsonify, request

from services.vertex_ai import about_ingredient, endpoint_predict_text2
from utils.translator import translate_to_english

search_bp: Blueprint = Blueprint("search", __name__)


@search_bp.route('/search', methods=['POST'])
def predict():
    try:
        # Retrieve the search keyword from the request data
        search_keyword = request.json.get('search_keyword', None)

        if not search_keyword:
            return jsonify({"error": "Search keyword is missing in the request"}), 400

        translate = translate_to_english(search_keyword)
        result = endpoint_predict_text2(translate)
        about = about_ingredient(translate)
        return jsonify({
            "success": "Search successful",
            "search_keyword": search_keyword,
            "result": result,
            "about": about
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
