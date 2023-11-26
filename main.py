import numpy as np
from flask import Flask, request
from PIL import Image

from app import MilvusConnection
from clip import encode_images, encode_text

app = Flask(__name__)

milvus_connection = MilvusConnection("image_embeddings")

@app.route('/photos', methods=['POST', 'DELETE'])
def upload_photos():
    if request.method == 'POST':
        file = request.files['image']
        file_name = file.filename
        image = Image.open(file)
        image_array = np.array(image)
        image_emb = encode_images(image_array)
        image_emb = image_emb.flatten().astype(float)
        image_id = milvus_connection.insert_image_data(file_name, image_emb)
        return image_id
    
    elif request.method == 'DELETE':
        image_id = request.args.get('image_id')
        milvus_connection.delete_image_data(image_id)
        return 'File deleted successfully!'


def extract_ids(hits):
    ids = []
    for hit in hits[0]:
        ids.append(hit.id)
    return ids


@app.route('/query', methods=['GET'])
def retrieve_photo():
    data = request.get_json()
    query = data.get('query')
    query_embd = encode_text(query)

    result = milvus_connection.search(query_embd)
    ids = extract_ids(result)
    
    return ids



if __name__ == '__main__':
    app.run(debug=True)
