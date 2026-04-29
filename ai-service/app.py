from flask import Flask, request, jsonify
from services.security import validate_input, sanitize
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# ✅ THEN register blueprints
app.register_blueprint(health_bp)
app.register_blueprint(query_bp)
app.register_blueprint(report_bp)

# Optional health check route
@app.route("/")
def home():
    return "Server is running"

if __name__ == "__main__":
    print("Flask is starting...")

    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
