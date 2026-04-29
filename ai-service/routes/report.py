from flask import Blueprint, request, jsonify
from threading import Thread
from services.job_store import create_job, update_job, get_job
from services.groq_client import GroqClient

report_bp = Blueprint("report", __name__)
client = GroqClient()


def generate_report_async(job_id, question):
    prompt = f"""
Generate a detailed risk report for:

{question}

Include:
- Summary
- Risk Type
- Recommendations
"""

    response = client.generate(prompt)

    result = response.get("result") if isinstance(response, dict) else response

    update_job(job_id, result)


@report_bp.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question required"}), 400

    job_id = create_job()

    thread = Thread(target=generate_report_async, args=(job_id, question))
    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "processing"
    })


@report_bp.route("/report-status/<job_id>", methods=["GET"])
def report_status(job_id):
    job = get_job(job_id)

    if not job:
        return jsonify({"error": "Invalid job_id"}), 404

    return jsonify(job)