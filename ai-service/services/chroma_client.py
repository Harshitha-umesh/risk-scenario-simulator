import chromadb
from sentence_transformers import SentenceTransformer

# Lazy loading (fixes your issue)
model = None
collection = None


def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def get_collection():
    global collection
    if collection is None:
        client = chromadb.Client()
        collection = client.create_collection(name="risk_docs")
    return collection


def add_documents(texts):
    model = get_model()
    collection = get_collection()

    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(texts))]
    )


def query_documents(query):
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    return results["documents"]
