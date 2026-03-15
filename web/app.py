from flask import Flask, render_template, request, send_file
from PIL import Image
import io

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    if file:
        img = Image.open(file.stream) # open the image from the uploaded file
    
        # TODO: Pass the image to the model for inference

        buf = io.BytesIO()
        img.save(buf, format="PNG") # write the image into the buffer
        buf.seek(0) # rewind to the start
        return send_file(buf, mimetype="image/png")
    
    return "No file uploaded"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
request.files["image"] — the key "image" must match what you used in formData.append()
io.BytesIO() — an in-memory buffer (so you don't have to save the image to disk)
buf.seek(0) — rewinds the buffer back to the start before sending it
send_file() — sends the image bytes back to the browser
"""