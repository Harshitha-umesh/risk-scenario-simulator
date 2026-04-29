from services.chroma_client import add_documents, query_documents

docs = [
    "Cyber attacks can cause data breaches",
    "Financial risks include market loss",
    "Operational risks affect daily processes"
]

add_documents(docs)

result = query_documents("cyber security risk")

print(result)