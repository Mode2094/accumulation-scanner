from flask import Flask, send_file
import os

app = Flask(__name__)

@app.get("/latest")
def latest_file():
    files = [f for f in os.listdir(".") if f.endswith(".json")]
    if not files:
        return {"error": "No files found"}, 404
    
    latest = max(files, key=os.path.getctime)
    return send_file(latest)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
