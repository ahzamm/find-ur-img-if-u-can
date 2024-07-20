import uuid

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sympy import false


class MilvusConnection:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.connect()
        self.create_schema()
        self.create_collection()
        self.create_index()

    def connect(self):
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to milvus server...")

    def create_schema(self):
        fields = [
            FieldSchema(
                name="pk", dtype=DataType.VARCHAR, max_length=200, is_primary=True
            ),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=8),
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="image_embeddings", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]
        self.schema = CollectionSchema(fields, enable_dynamic_field=True)

    def delete_schema(self):
        utility.drop_collection(self.collection_name)

    def create_collection(self):
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name, self.schema)
        else:
            self.collection = Collection(self.collection_name)

    def create_index(self):
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.collection.create_index("image_embeddings", index)

    def insert_image_data(self, user_id, image_name, image_emb, pk_id):
        if pk_id is not None:
            data = [[pk_id], [user_id], [image_name], [image_emb]]
        else:
            data = [[self.generate_id()], [user_id], [image_name], [image_emb]]
        result = self.collection.insert(data)
        self.collection.flush()
        self.collection.load()
        image_id = result.primary_keys[0]
        return image_id

    def delete_image_data(self, image_id):
        expr = f"pk in {[image_id]}"
        self.collection.delete(expr)
        self.collection.load()

    def disconnect(self):
        connections.disconnect(alias="default")

    def generate_id(self, length=8):
        uuid_str = str(uuid.uuid4()).replace("-", "")
        short_id = uuid_str[:length]
        return short_id

    def checkVectorId(self, id):
        expr = f"pk == '{id}'"
        results = self.collection.query(expr)
        return len(results) > 0

    def search(self, user_id, query_embd, similarity_threshold=0.4, top_k=100):
        try:
            self.collection.load()
        except Exception as e:
            return []

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        query_embd = query_embd.astype(float)
        expr = f"user_id == '{user_id}'"

        try:
            results = self.collection.search(
                query_embd, "image_embeddings", search_params, top_k, expr
            )
        except Exception as e:
            return []

        filtered_results = []
        for result in results:
            for hit in result:
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + hit.distance)
                if similarity > similarity_threshold:
                    filtered_results.append(
                        {
                            "id": hit.id,
                            "distance": hit.distance,
                            "similarity": similarity,
                        }
                    )

        return filtered_results

    def get_all_photos(self, user_id):
        expr = f"user_id == '{user_id}'"
        results = self.collection.query(expr)
        photos = [{"vector_id": result["pk"]} for result in results]
        return photos
