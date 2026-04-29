from flask import Blueprint, request, jsonify
from services.groq_client import call_groq
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
analyse_document_bp = Blueprint("analyse_document", __name__)

@analyse_document_bp.route("/analyse-document", methods=["POST"])
def analyse_document():
    data = request.get_json()
    if not data or not data.get("input"):
        return jsonify({"error": "input field is required"}), 400
    user_input = data["input"].strip()
    if len(user_input) < 20:
        return jsonify({"error": "input too short"}), 400
    with open(os.path.join(BASE_DIR, "prompts", "analyse_document.txt"), "r") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{input}", user_input)
    result = call_groq(prompt, temperature=0.3, max_tokens=1500)
    if result is None:
        return jsonify({
            "error": "AI service unavailable",
            "is_fallback": True,
            "findings": [],
            "summary": "Analysis failed. Please try again.",
            "total_findings": 0,
            "risk_count": 0,
            "insight_count": 0
        }), 503
    try:
        return jsonify(json.loads(result)), 200
    except Exception:
        return jsonify({"raw_response": result}), 200