from flask import Blueprint, request, jsonify
from services.groq_client import call_groq
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or not data.get("input"):
        return jsonify({"error": "input field is required"}), 400
    user_input = data["input"].strip()
    if len(user_input) < 10:
        return jsonify({"error": "input too short"}), 400
    with open(os.path.join(BASE_DIR, "prompts", "recommend.txt"), "r") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{input}", user_input)
    result = call_groq(prompt, temperature=0.7)
    if result is None:
        return jsonify({"error": "AI service unavailable", "is_fallback": True}), 503
    try:
        return jsonify(json.loads(result)), 200
    except Exception:
        return jsonify({"raw_response": result}), 200