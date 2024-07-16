from datetime import datetime
import sys
import numpy as np
from flask import Flask, request
from PIL import Image

from milvus import MilvusConnection
from clip import encode_images, encode_text

app = Flask(__name__)


milvus_connection = MilvusConnection("image_embeddings")


@app.route("/photos", methods=["POST", "DELETE"])
def upload_photos():
    try:
        if request.method == "POST":
            user_id = request.form["user_id"]
            file = request.files["image"]
            pk_id = request.form["vector_id"]
            file_name = file.filename
            image = Image.open(file)
            image_array = np.array(image)
            image_emb = encode_images(image_array)
            image_emb = image_emb.flatten().astype(float)
            vector_id = milvus_connection.insert_image_data(
                user_id, file_name, image_emb, pk_id
            )
            return {"success": "true", "vector_id": vector_id}

        elif request.method == "DELETE":
            image_id = request.args.get("image_id")
            milvus_connection.delete_image_data(image_id)
            return {"success": "true", "message": "File deleted successfully!"}

    except Exception as e:
        return {"success": "false", "error": str(e)}, 500


def extract_ids(hits):
    ids = []
    for hit in hits:
        ids.append(hit.get("id"))
    return ids


@app.route("/query", methods=["POST"])
def retrieve_photo():
    try:
        data = request.get_json()
        query = data.get("query")
        user_id = data.get("user_id")
        query_embd = encode_text(query)

        result = milvus_connection.search(user_id, query_embd)
        ids = extract_ids(result)

        return {"success": "true", "image_ids": ids}

    except Exception as e:
        return {"success": "false", "error": str(e)}, 500


@app.route("/delete-schema", methods=["DELETE"])
def delete_schema():
    try:
        milvus_connection.delete_schema()
        return {"success": "true", "message": "Schema deleted successfully"}
    except Exception as e:
        return {"success": "false", "message": str(e)}, 500

@app.route("/create-schema", methods=["GET"])
def create_schema():
    try:
        milvus_connection.create_collection()
        return {"success": "true", "message": "Schema created successfully"}
    except Exception as e:
        return {"success": "false", "message": str(e)}, 500


@app.route("/check-vector", methods=["GET"])
def check_ids():
    try:
        vector_id = request.args.get("vector_id")
        if not vector_id:
            return {"success": "false", "error": "vector_id is required"}, 400

        exists = milvus_connection.checkVectorId(vector_id)
        return {"success": "true", "exists": exists}
    except Exception as e:
        return {"success": "false", "error": str(e)}, 500


if __name__ == "__main__":
    app.run(port=5001, debug=True)
