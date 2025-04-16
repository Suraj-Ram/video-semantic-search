from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, connections

connections.connect(host='localhost', port='19530')
print("Connected to Milvus server at port", '19530')


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64,
                    descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                    descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='video retrieval')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',  # IP
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection
