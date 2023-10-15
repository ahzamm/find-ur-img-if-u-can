import time

import numpy as np
from PIL import Image
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from clip import encode_images



connections.connect(alias="default", host='localhost', port='19530')
print("Connected to milvus server...")



fields = [FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
          FieldSchema(name="image_embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)]
schema = CollectionSchema(fields, enable_dynamic_field=True)



collection_name = "image_embeddings"
milvus_connection = Collection(collection_name, schema) if not utility.has_collection(collection_name) else Collection(collection_name)



index = {"index_type": "IVF_FLAT", "metric_type": "L2", "params":{"nlist":128}}
milvus_connection.create_index("image_embeddings", index)

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.event_type == 'created':
            image = Image.open(event.src_path)
            image_array = np.array(image)
            image_emb = encode_images(image_array)
            image_emb = image_emb.flatten().astype(float)
            data = [[image_emb]]
            milvus_connection.insert(data)
            milvus_connection.flush()


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