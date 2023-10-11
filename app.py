import time

import numpy as np
from PIL import Image
from pymilvus import connections, utility
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from clip import encode_images

connections.connect(alias="default", host='localhost', port='19530')

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