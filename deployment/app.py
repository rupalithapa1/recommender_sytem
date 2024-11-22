from flask import Flask, render_template, request, flash, redirect, url_for, session

from deployment.pipeline.prediction_pipeline import PredictionPipeline
from werkzeug.utils import secure_filename
import os
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import cv2
import numpy as np


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add secret key for flashing

# Allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the uploaded file is an allowed image format
def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            # Read the file as bytes and convert to NumPy array
            img_bytes = np.frombuffer(file.read(), np.uint8)
            
            # Decode the image with OpenCV
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            # Now pass the image through the prediction pipeline
            prediction_pipeline = PredictionPipeline()
            predicted_label = prediction_pipeline.predict_label(img)  # Pass the image

            return render_template("predict.html", label=predicted_label)
        else:
            flash('File is not an image or invalid format. Please upload a .png, .jpg, or .jpeg file.')
            return redirect(request.url)

    return render_template("upload_image.html")


if __name__ == "__main__":
    app.run(port=5000)