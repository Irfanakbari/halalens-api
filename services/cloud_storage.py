import os

from google.auth import credentials
from google.cloud import storage

BUCKET_NAME = os.environ.get('BUCKET_NAME')
storage_client = storage.Client()
# /Users/irfanakbari/Downloads/halalens-410510-3f7fe2d03cb4.json


def upload_to_storage(file, uid):
    # Create a bucket object
    bucket = storage_client.get_bucket(BUCKET_NAME)

    # Specify the destination path in the bucket (using user's UID)
    destination_blob_name = f"{uid}/{file.filename}"

    # Create a blob object (representing the file in the bucket)
    blob = bucket.blob(destination_blob_name)

    # Upload the file to the bucket
    blob.upload_from_file(file)

    # Generate the URI for the uploaded file
    uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"

    return uri
