from flask import Flask, jsonify
from ml_models import compare_models

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Learning Companion Flask API is running!"})

@app.route("/compare", methods=["GET"])
def compare():
    """Return model accuracy comparison as JSON"""
    results = compare_models()
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
