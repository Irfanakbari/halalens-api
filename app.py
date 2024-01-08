import ssl
from firebase_admin import initialize_app
from flask import Flask

from middlewares.auth_middleware import verify_firebase_id_token, verify_app_check
from routes import register_blueprints

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
initialize_app()

app = Flask(__name__)
register_blueprints(app)
app.before_request(verify_firebase_id_token)
app.before_request(verify_app_check)

if __name__ == '__main__':
    app.run(debug=False, port=8080)
