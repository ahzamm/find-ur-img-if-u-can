import numpy as np
from flask import Flask, request
from PIL import Image

from app import MilvusConnection
from clip import encode_images

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


@app.route('/query', methods=['GET'])
def retrieve_photo():
    data = request.get_json()
    query = data.get('query')
    print("🚀  main.py:33 query :", query)
    return query



if __name__ == '__main__':
    app.run(debug=True)
