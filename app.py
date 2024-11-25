from flask import Flask, render_template, request, redirect, flash
from deployment.pipeline.prediction_pipeline import PredictionPipeline  # Import your pipeline

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flashing messages

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend")
def recommend():
    return render_template("recommend.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        ratings = request.form.get("ratings")
        
        try:
            # Convert ratings to a list of numbers
            ratings_list = [float(r.strip()) for r in ratings.split(",")]

            # Use PredictionPipeline to generate recommendations
            prediction_pipeline = PredictionPipeline()
            predictions = prediction_pipeline.predict(user_id, ratings_list)  # Example function

            # Render results
            return render_template("result.html", user_id=user_id, ratings=ratings, predictions=predictions)
        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect("/recommend")

if __name__ == "__main__":
    app.run(port=5000)
