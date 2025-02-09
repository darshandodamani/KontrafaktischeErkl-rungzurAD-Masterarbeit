from flask import Flask, render_template, request, redirect, url_for
import csv
import random
import os
import time

app = Flask(__name__)

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path for responses.csv inside the project directory
responses_file = os.path.join(BASE_DIR, "responses.csv")

# Image dataset with labels
image_data = [
    {"original": "original1.png", "method1": "method1_cf1.png", "method2": "method2_cf1.png", "method3": "method3_cf1.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},

    {"original": "original2.png", "method1": "method1_cf2.png", "method2": "method2_cf2.png", "method3": "method3_cf2.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},

    {"original": "original3.png", "method1": "method1_cf3.png", "method2": "method2_cf3.png", "method3": "method3_cf3.png",
     "labels": {"original": "STOP", "method1": "STOP", "method2": "GO", "method3": "GO"}},

    {"original": "original4.png", "method1": "method1_cf4.png", "method2": "method2_cf4.png", "method3": "method3_cf4.png",
     "labels": {"original": "STOP", "method1": "STOP", "method2": "GO", "method3": "GO"}},
]


# Ensure responses file exists and write headers if it's new
def initialize_csv():
    if not os.path.exists(responses_file) or os.stat(responses_file).st_size == 0:
        with open(responses_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Chosen Method", "Interpretability", "Plausibility", "Actionability", "Trust in AI", "Visual Coherence", "Comments"])

# Call function to ensure CSV is properly initialized
initialize_csv()

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    if request.method == "POST":
        # Print received form data for debugging
        print("Form Data Received:", request.form.to_dict())

        # Get form data safely
        image = request.form.get("image", "").strip()
        preferred_method = request.form.get("preferred_method", "").strip()
        interpretability = request.form.get("interpretability", "").strip()
        plausibility = request.form.get("plausibility", "").strip()
        actionability = request.form.get("actionability", "").strip()
        trust_ai = request.form.get("trust_ai", "").strip()
        visual_coherence = request.form.get("visual_coherence", "").strip()
        comments = request.form.get("comments", "").strip()  # Allow empty comments

        # Check if any required field is missing
        missing_fields = []
        if not image:
            missing_fields.append("image")
        if not preferred_method:
            missing_fields.append("preferred_method")
        if not interpretability:
            missing_fields.append("interpretability")
        if not plausibility:
            missing_fields.append("plausibility")
        if not actionability:
            missing_fields.append("actionability")
        if not trust_ai:
            missing_fields.append("trust_ai")
        if not visual_coherence:
            missing_fields.append("visual_coherence")

        if missing_fields:
            print("🚨 Missing fields:", missing_fields)  # Debugging print
            return f"Error: Missing form fields: {', '.join(missing_fields)}", 400

        # Save response inside correct directory
        with open(responses_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([image, preferred_method, interpretability, plausibility, actionability, trust_ai, visual_coherence, comments])

        return redirect(url_for("thank_you"))

    # Select a random image
    selected_image = random.choice(image_data)
    return render_template("index.html", image_data=selected_image, timestamp=int(time.time()))


@app.route("/thank_you")
def thank_you():
    return render_template("thank_you.html")

if __name__ == "__main__":
    app.run(debug=True)
