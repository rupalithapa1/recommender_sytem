from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for flashing messages

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Dummy Recommendation Pipeline for demonstration
class RecommendationPipeline:
    def predict(self, user_id, data):
        # Example: Just return dummy recommendations based on User ID
        return [f"Item {i} for User {user_id}" for i in range(1, 6)]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get User ID
        user_id = request.form.get("user_id")
        
        # Get uploaded file
        file = request.files.get("file")
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Read the file into a DataFrame
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                flash("Invalid file format or corrupted data.")
                return redirect(url_for("index"))

            # Process data with prediction pipeline
            prediction_pipeline = RecommendationPipeline()
            recommendations = prediction_pipeline.predict(user_id, data)

            # Render the results
            return render_template("predict.html", user_id=user_id, recommendations=recommendations)
        else:
            flash("Please upload a valid CSV file.")
            return redirect(url_for("index"))

# Create 'uploads' directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
