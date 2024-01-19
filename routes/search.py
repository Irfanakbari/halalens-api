import flask
from flask import Blueprint, jsonify, request

from services.vertex_ai import search_ingredient

search_bp: Blueprint = Blueprint("search", __name__)


@search_bp.route('/search', methods=['POST'])
def predict():
    try:
        # Retrieve the search keyword from the request data
        search_keyword = request.json.get('search_keyword', None)

        if not search_keyword:
            return jsonify({"error": "Search keyword is missing in the request"}), 400

        result = search_ingredient(search_keyword)
        return jsonify({
                "success": "Search successful",
                "search_keyword": search_keyword,
                "result": result
            }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
