from flask import Flask, render_template, request, send_file
import sys
from pathlib import Path
# Add project root to path so Flask can find sibling packages
sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.inference_image import uploaded_image_to_tensor, inference_single_image, tensor_to_png_buffer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    if file:
        image_tensor = uploaded_image_to_tensor(file)

        # Run inference
        boxed_image = inference_single_image(image_tensor) 

        buf = tensor_to_png_buffer(boxed_image)

        return send_file(buf, mimetype="image/png") # sends the image bytes back to the browser
    
    return "No file uploaded"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)