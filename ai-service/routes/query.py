from flask import Blueprint, request, jsonify
from services.groq_client import GroqClient
from services.chroma_client import query_documents
from services.redis_client import get_cache, set_cache
import json

query_bp = Blueprint("query", __name__)
client = GroqClient()

@query_bp.route("/query", methods=["POST"])
def query():
    try:
        question = request.json.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # STEP 0: CACHE CHECK
        cache_key = f"query:{question}"
        cached = get_cache(cache_key)

        if cached:
            return jsonify({
                "result": json.loads(cached),
                "cached": True
            })

        # Step 1: Get context
        docs = query_documents(question)
        context = " ".join(docs[0]) if docs else ""

        # Step 2: Prompt
        prompt = f"""
You are a professional Risk Analysis AI.

Use ONLY the given context.
Answer in 2-3 lines.

Context:
{context}

Question:
{question}

Return ONLY JSON:
{{
  "answer": "...",
  "risk_type": "Financial | Operational | Security | Technical",
  "confidence": 0.0
}}
"""

        # Step 3: AI call
        ai_response = client.generate(prompt)

        response = ai_response.get("result") if isinstance(ai_response, dict) else ai_response

        # Step 4: Fallback
        if not response:
            response = {
                "answer": "Unable to process",
                "risk_type": "Unknown",
                "confidence": 0.5
            }

        # Step 5: Convert
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except:
                response = {
                    "answer": response,
                    "risk_type": "Unknown",
                    "confidence": 0.6
                }

        # STEP 6: SAVE CACHE
        set_cache(cache_key, json.dumps(response))

        return jsonify({
            "result": response,
            "sources": docs,
            "meta": ai_response.get("meta", {}),
            "cached": False
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500