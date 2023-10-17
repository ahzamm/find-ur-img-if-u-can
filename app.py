import os
import time
import uuid

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
        self.id = self.generate_id()
        self.connect()
        self.create_schema()
        self.create_collection()
        self.create_index()

    def connect(self):
        connections.connect(alias="default", host='localhost', port='19530')
        print("Connected to milvus server...")

    def create_schema(self):
        fields = [FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=8, is_primary=True),
                  FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=20),
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

    def insert_image_data(self, image_name, image_emb):
        data = [[self.id], [image_name], [image_emb]]
        self.collection.insert(data)
        self.collection.flush()
        self.collection.load()

    def disconnect(self):
        connections.disconnect(alias="default")

    def generate_id(self, length=8):
        uuid_str = str(uuid.uuid4()).replace("-", "")
        short_id = uuid_str[:length]
        return short_id



class ImageHandler:
    def encode_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)
        image_emb = encode_images(image_array)
        return image_emb.flatten().astype(float)


class EventHandler(FileSystemEventHandler):
    def __init__(self, milvus_connection, image_handler):
        self.milvus_connection = milvus_connection
        self.image_handler = image_handler

    def on_created(self, event):
        if event.event_type == 'created':
            image_name = os.path.basename(event.src_path)
            image_emb = self.image_handler.encode_image(event.src_path)
            self.milvus_connection.insert_image_data(image_name, image_emb)
            print("Embd stored in database successfully...")



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
