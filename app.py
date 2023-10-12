import time

import numpy as np
from PIL import Image
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from clip import encode_images

connections.connect(alias="default", host='localhost', port='19530')


fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="image_embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)
]
schema = CollectionSchema(fields, "Store the image embds for image search engine")


collection_name = "image_embeddings"
if not utility.has_collection(collection_name):
    milvus_connection = Collection(collection_name, schema)
else:
    milvus_connection = Collection(collection_name)


try:
    print(utility.get_server_version())
except Exception as e:
    print("Failed to connect to Milvus server:", e)


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.event_type == 'created':
            image = Image.open(event.src_path)
            image_array = np.array(image)
            image_emb = encode_images(image_array)
            print('Lowest embeding value: ', image_emb.min(), 'Heighest embeding value: ', image_emb.max())



if __name__ == "__main__":
    path = "./images"
    event_handler = MyHandler()
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


connections.disconnect()