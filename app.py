import os
import ssl

from dotenv import load_dotenv
from firebase_admin import initialize_app
from flask import Flask
from google.cloud import aiplatform

from middlewares.auth_middleware import verify_firebase_id_token
from routes import register_blueprints
ssl._create_default_https_context = ssl._create_unverified_context

initialize_app()

# Load environment variables from a file
load_dotenv()


app = Flask(__name__)
register_blueprints(app)
app.before_request(verify_firebase_id_token)
# app.before_request(verify_app_check)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
