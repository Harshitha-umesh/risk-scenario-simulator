from flask import Blueprint, request, jsonify
from services.groq_client import call_groq
from services.chroma_client import query_knowledge
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    try:
        question = request.json.get("question")

        logger.info(f"Received question: {question}")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # ✅ CREATE CLIENT HERE (NOT GLOBAL)
        client = GroqClient()

        # ✅ CACHE CHECK
        cache_key = f"query:{question}"
        cached = get_cache(cache_key)

        if cached:
            logger.info("Cache hit")
            return jsonify({
                "result": json.loads(cached),
                "cached": True,
                "meta": {"is_fallback": False}
            })

        # ✅ NO CHROMA
        docs = []
        context = ""

        # ✅ PROMPT
        prompt = f"""
You are a professional Risk Analysis AI.
Answer in 2-3 lines.

Question:
{question}

Return ONLY JSON:
{{
  "answer": "...",
  "risk_type": "Financial | Operational | Security | Technical",
  "confidence": 0.0
}}
"""

        try:
            logger.info("Calling AI model")

            ai_response = client.generate(prompt)

            response = ai_response.get("result")
            meta = ai_response.get("meta", {})

            is_fallback = meta.get("is_fallback", False)

        except Exception as e:
            logger.warning("AI failed, using fallback response")

            response = {
                "answer": "Potential risk detected. Further analysis recommended.",
                "risk_type": "General",
                "confidence": 0.5
            }

            meta = {"error": str(e)}
            is_fallback = True

        # ✅ HANDLE STRING RESPONSE
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except:
                response = {
                    "answer": response,
                    "risk_type": "Unknown",
                    "confidence": 0.6
                }

        # ✅ CACHE ONLY IF NOT FALLBACK
        if not is_fallback:
            set_cache(cache_key, json.dumps(response))

        logger.info("Response generated successfully")

        return jsonify({
            "result": response,
            "sources": docs,
            "meta": {
                **meta,
                "is_fallback": is_fallback
            },
            "cached": False
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
