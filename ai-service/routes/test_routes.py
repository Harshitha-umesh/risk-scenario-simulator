# Separate file for test-related routes
from flask import Blueprint, request, jsonify

# Create blueprint
test_bp = Blueprint("test", __name__)

# Test endpoint (used for middleware + limiter validation)
@test_bp.route("/test", methods=["POST"])
def test():
    return jsonify({
        "message": "passed",
        "input": getattr(request, "sanitized_input", None)
    })