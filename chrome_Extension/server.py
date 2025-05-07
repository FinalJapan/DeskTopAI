from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/browser-data', methods=['POST'])
def browser_data():
    data = request.json
    print("Received browser data:", data)
    return "OK", 200

if __name__ == "__main__":
    app.run(port=5000)