from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Sample response from OralSnap app!</p>"
