from flask import Blueprint, request, jsonify
from services.groq_client import GroqClient
from services.chroma_client import query_documents

query_bp = Blueprint("query", __name__)
client = GroqClient()


@query_bp.route("/query", methods=["POST"])
def query():
    try:
        question = request.json.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Step 1: Get context from ChromaDB
        docs = query_documents(question)

        context = " ".join(docs[0]) if docs else ""

        # Step 2: Send to AI
        prompt = f"""
You are a risk analysis assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly in 3-4 lines.
"""

        ai_response = client.generate(prompt)

        return jsonify({
            "answer": ai_response["result"],
            "sources": docs,
            "meta": ai_response["meta"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500