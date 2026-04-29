import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
 # model = SentenceTransformer('all-MiniLM-L6-v2')

# Create DB client
#client = chromadb.Client()

# Create collection
#collection = client.create_collection(name="risk_docs")


#def add_documents(texts):
   # embeddings = model.encode(texts).tolist()

    #collection.add(
        #documents=texts,
        #embeddings=embeddings,
        #ids=[str(i) for i in range(len(texts))]
    #)
def add_documents(texts):
    return


# def query_documents(query):
#     query_embedding = model.encode([query]).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=3
#     )

#     return results["documents"]
def query_documents(question):
    return []