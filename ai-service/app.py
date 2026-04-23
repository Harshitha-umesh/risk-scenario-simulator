from flask import Flask
from routes.categorise import categorise_bp
from routes.query import query_bp
from routes.health import health_bp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)

app.register_blueprint(categorise_bp)
app.register_blueprint(query_bp)
app.register_blueprint(health_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)