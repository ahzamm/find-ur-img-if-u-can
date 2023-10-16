import os
import time

import numpy as np
from PIL import Image
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from clip import encode_images

class MilvusConnection:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.connect()
        self.create_schema()
        self.create_collection()
        self.create_index()

    def connect(self):
        connections.connect(alias="default", host='localhost', port='19530')
        print("Connected to milvus server...")

    def create_schema(self):
        fields = [FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
                  FieldSchema(name="image_embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)]
        self.schema = CollectionSchema(fields, enable_dynamic_field=True)

    def create_collection(self):
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name, self.schema)
        else:
            self.collection = Collection(self.collection_name)

    def create_index(self):
        index = {"index_type": "IVF_FLAT", "metric_type": "L2", "params":{"nlist":128}}
        self.collection.create_index("image_embeddings", index)

    def insert_image_embedding(self, image_emb):
        data = [[image_emb]]
        result = self.collection.insert(data)
        primary_key = result.primary_keys[0]
        self.collection.flush()
        self.collection.load()
        return primary_key

    def disconnect(self):
        connections.disconnect()



class ImageHandler:
    def encode_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)
        image_emb = encode_images(image_array)
        return image_emb.flatten().astype(float)

    def rename_image(self, original_path, primary_key):
        folder_path, file_name = os.path.split(original_path)
        new_file_name = f"{primary_key}.png"
        new_path = os.path.join(folder_path, new_file_name)
        os.rename(original_path, new_path)



class EventHandler(FileSystemEventHandler):
    def __init__(self, milvus_connection, image_handler):
        self.milvus_connection = milvus_connection
        self.image_handler = image_handler

    def on_created(self, event):
        if event.event_type == 'created':
            image_emb = self.image_handler.encode_image(event.src_path)
            primary_key = self.milvus_connection.insert_image_embedding(image_emb)
            print("Embd stored in database successfully...")

            self.image_handler.rename_image(event.src_path, primary_key)
            print("Image successfully renamed to its pk")



if __name__ == "__main__":
    path = "./images"

    milvus_connection = MilvusConnection("image_embeddings")
    image_handler = ImageHandler()
    event_handler = EventHandler(milvus_connection, image_handler)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    print("Watching for changes in the 'images' folder...")

    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

    milvus_connection.disconnect()
