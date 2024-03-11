import logging
from datetime import datetime

import numpy as np
from flask import Flask, request
from PIL import Image

from app import MilvusConnection
from clip import encode_images, encode_text


app = Flask(__name__)

logging.basicConfig(filename="logs/app.log", level=logging.ERROR)

milvus_connection = MilvusConnection("image_embeddings")


def log_error(exception):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"{current_time} - An error occurred: {exception}"
    logging.error(error_message)
    print(error_message)


@app.route("/photos", methods=["POST", "DELETE"])
def upload_photos():
    try:
        if request.method == "POST":
            file = request.files["image"]
            file_name = file.filename
            image = Image.open(file)
            image_array = np.array(image)
            image_emb = encode_images(image_array)
            image_emb = image_emb.flatten().astype(float)
            image_id = milvus_connection.insert_image_data(file_name, image_emb)
            return {"success": "true", "image_id": image_id}

        elif request.method == "DELETE":
            image_id = request.args.get("image_id")
            milvus_connection.delete_image_data(image_id)
            return {"success": "true", "message": "File deleted successfully!"}

    except Exception as e:
        log_error(e)
        return {"success": "false", "error": str(e)}, 500


def extract_ids(hits):
    ids = []
    for hit in hits[0]:
        ids.append(hit.id)
    return ids


@app.route("/query", methods=["GET"])
def retrieve_photo():
    try:
        data = request.get_json()
        query = data.get("query")
        query_embd = encode_text(query)

        result = milvus_connection.search(query_embd)
        ids = extract_ids(result)

        return {"success": "true", "image_ids": ids}

    except Exception as e:
        log_error(e)
        return {"success": "false", "error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True)
