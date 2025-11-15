from flask import Flask, send_file
import os

app = Flask(__name__)

@app.get("/")
def home():
    return "Accumulation Scanner Running"

@app.get("/latest")
def latest_file():
    files = [f for f in os.listdir(".") if f.endswith(".json")]
    if not files:
        return {"error": "No JSON files found yet"}, 404

    latest = max(files, key=os.path.getctime)
    return send_file(latest, mimetype="application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
