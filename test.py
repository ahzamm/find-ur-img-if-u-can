from pymilvus import FieldSchema, DataType, CollectionSchema, Collection

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="image_embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]
schema = CollectionSchema(fields, "Store the image embds for image search engine")
hello_milvus = Collection("image_search_db", schema)