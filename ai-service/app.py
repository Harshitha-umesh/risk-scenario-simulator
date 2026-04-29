from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute"]
)

# Register blueprints
from routes.describe import describe_bp
from routes.recommend import recommend_bp
from routes.generate_report import generate_report_bp
from routes.query import query_bp
from routes.generate_report_stream import generate_report_stream_bp
from routes.analyse_document import analyse_document_bp

app.register_blueprint(describe_bp)
app.register_blueprint(recommend_bp)
app.register_blueprint(generate_report_bp)
app.register_blueprint(query_bp)
app.register_blueprint(generate_report_stream_bp)
app.register_blueprint(analyse_document_bp)

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile",
        "service": "Risk Scenario Simulator AI"
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)