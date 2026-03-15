from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/hello", methods=["POST"])
def hello():
    data = request.json
    name = data.get("name")

    return jsonify({"message": f"Hello {name}!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)