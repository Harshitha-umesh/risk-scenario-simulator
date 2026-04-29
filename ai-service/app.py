from flask import Flask
from routes.health import health_bp
from routes.query import query_bp
from routes.report import report_bp


# ✅ FIRST create app
app = Flask(__name__)

# ✅ THEN register blueprints
app.register_blueprint(query_bp)
app.register_blueprint(health_bp)
app.register_blueprint(report_bp)

@app.route("/")
def home():
    return "Server working ✅"

if __name__ == "__main__":
    print("Flask is starting...")
    app.run(debug=True, port=5000, use_reloader=False)