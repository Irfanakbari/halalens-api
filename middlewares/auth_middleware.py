import flask
import jwt
from firebase_admin import app_check
from firebase_admin.auth import verify_id_token


def verify_firebase_token(token):
    try:
        # Verify Firebase ID token
        decoded_token = verify_id_token(token)
        return decoded_token
    except ValueError as e:
        print(f"Token verification failed: {e}")
        return None


def verify_firebase_id_token() -> None:
    # Extract the Firebase ID token from the Authorization header
    authorization_header = flask.request.headers.get('Authorization', default="")
    if not authorization_header or not authorization_header.startswith('Bearer '):
        flask.abort(401)
    id_token = authorization_header.split(' ')[1]
    if not authorization_header:
        flask.abort(401)
    # Verify the Firebase ID token
    decoded_token = verify_firebase_token(id_token)
    if not decoded_token:
        flask.abort(401)

    # Add the decoded token to the request context for later use if needed
    flask.g.decoded_token = decoded_token


def verify_app_check() -> None:
    app_check_token = flask.request.headers.get("X-Firebase-AppCheck", default="")
    try:
        app_check_claims = app_check.verify_token(app_check_token)
        # If verify_token() succeeds, okay to continue to route handler.
    except (ValueError, jwt.exceptions.DecodeError):
        flask.abort(401)
