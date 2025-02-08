import ssl

from dotenv import load_dotenv
from flask import Flask
from routes import register_blueprints

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from a file
load_dotenv()


app = Flask(__name__)
register_blueprints(app)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
