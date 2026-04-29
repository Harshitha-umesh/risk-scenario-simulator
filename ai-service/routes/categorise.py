from flask import Blueprint, request, jsonify
from services.groq_client import GroqClient
import json

categorise_bp = Blueprint("categorise", __name__)
client = GroqClient()


@categorise_bp.route("/categorise", methods=["POST"])
def categorise():
    try:
        user_input = request.json.get("text")

        if not user_input:
            return jsonify({"error": "Text input is required"}), 400

        prompt = f"""
You are an AI risk classifier.

Classify the following risk into ONE category:
- Financial
- Operational
- Security
- Technical

Also provide:
- confidence (0 to 1)
- short reasoning

Return ONLY valid JSON in this format:
{{
  "category": "",
  "confidence": 0.0,
  "reasoning": ""
}}

Text: {user_input}
"""

        ai_response = client.generate(prompt)

        # Try to convert string → JSON
        try:
            parsed = json.loads(ai_response["result"])
        except:
            parsed = {
                "category": "Unknown",
                "confidence": 0.0,
                "reasoning": "Parsing failed"
            }

        return jsonify({
            "data": parsed,
            "meta": ai_response["meta"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500