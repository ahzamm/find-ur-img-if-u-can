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
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        
        # Check if the collection exists and is not empty before creating the index
        if not utility.has_collection(self.collection_name):
            raise Exception(f"Collection {self.collection_name} does not exist")
        
        # Make sure the collection has data before creating the index
        self.collection.flush()
        
        # Create the index and load the collection
        self.collection.create_index("image_embeddings", index_params)
        self.collection.load()
        print("Index created and collection loaded.")


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

    def search(self, user_id, query_embd, similarity_threshold=0.2, top_k=100):
        self.ensure_collection_loaded() 
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
        # Ensure the collection is loaded
        if not self.collection.is_empty:
            self.collection.load()
        
        # Check if the index exists
        index_exists = self.collection.has_index()
        if not index_exists:
            raise Exception("Index does not exist for the collection")

        expr = f"user_id == '{user_id}'"
        results = self.collection.query(expr)
        photos = [{"vector_id": result["pk"]} for result in results]
        
        return photos

    
    def delete_user_vectors(self, user_id):
        try:
            # Ensure the collection is loaded
            self.collection.load()

            # Step 1: Query to get all primary keys (pk) associated with the user_id
            expr = f"user_id == '{user_id}'"
            results = self.collection.query(expr, output_fields=["pk"])
            
            # Extract the primary keys from the query result
            primary_keys = [result["pk"] for result in results]
            
            if not primary_keys:
                print(f"No vectors found for user {user_id}.")
                return {"success": "false", "message": f"No vectors found for user {user_id}"}
            
            # Step 2: Delete the vectors using the primary keys
            delete_expr = f"pk in {primary_keys}"
            delete_result = self.collection.delete(delete_expr)

            # Step 3: Flush the collection to make sure the deletion is applied
            self.collection.flush()
            
            # Step 4: Reload the collection to reflect changes
            self.collection.load()
            
            print(f"Deleted vectors for user {user_id}: {delete_result}")
            return {"success": "true", "message": f"Deleted vectors for user {user_id}"}
        
        except Exception as e:
            print(f"Error deleting vectors for user {user_id}: {str(e)}")
            raise



    def ensure_collection_loaded(self):
        if not self.collection.is_empty:
            self.collection.load()
        else:
            raise Exception("Collection is empty or not available")



