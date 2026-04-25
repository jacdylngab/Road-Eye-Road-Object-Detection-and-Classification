from flask import Flask, render_template, request, send_file
import io

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("video.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")

    if not file:
        return "No file selected"

    video_stream = io.BytesIO(file.read())
    video_stream.seek(0) # After writing to the stream, the pointer is at the end. We need to go back at the beginning.
    return send_file(video_stream, mimetype="video/mp4")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)