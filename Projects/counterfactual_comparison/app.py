from flask import Flask, render_template, request, redirect, url_for, flash
import csv
import random
import os
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Get the current script directory and CSV file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
responses_file = os.path.join(BASE_DIR, "responses.csv")

# Image dataset with labels
image_data = [
    {"original": "original1.png",
     "method1": "method1_cf1.png",
     "method2": "method2_cf1.png",
     "method3": "method3_cf1.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},
    {"original": "original2.png",
     "method1": "method1_cf2.png",
     "method2": "method2_cf2.png",
     "method3": "method3_cf2.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},
    {"original": "original3.png",
     "method1": "method1_cf3.png",
     "method2": "method2_cf3.png",
     "method3": "method3_cf3.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},
    {"original": "original4.png",
     "method1": "method1_cf4.png",
     "method2": "method2_cf4.png",
     "method3": "method3_cf4.png",
     "labels": {"original": "STOP", "method1": "GO", "method2": "GO", "method3": "GO"}},
]

# Update CSV header – one row per evaluation that now includes 4 fields for each counterfactual image.
def initialize_csv():
    header = [
        "Timestamp",
        "Image",
        "Interpretability_method1", "Plausibility_method1", "Visual_Coherence_method1", "Comment_method1",
        "Interpretability_method2", "Plausibility_method2", "Visual_Coherence_method2", "Comment_method2",
        "Interpretability_method3", "Plausibility_method3", "Visual_Coherence_method3", "Comment_method3",
    ]
    if not os.path.exists(responses_file) or os.stat(responses_file).st_size == 0:
        with open(responses_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

# Ensure CSV file is initialized with the header.
initialize_csv()

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    if request.method == "POST":
        # Retrieve the original image identifier (hidden field)
        original_image = request.form.get("image", "").strip()
        methods = ["method1", "method2", "method3"]
        ratings = {}

        missing_fields = []
        # For each method, get the three ratings (required) and one comment (optional)
        for method in methods:
            interpretability = request.form.get(f"interpretability_{method}", "").strip()
            plausibility = request.form.get(f"plausibility_{method}", "").strip()
            visual_coherence = request.form.get(f"visual_coherence_{method}", "").strip()
            comment = request.form.get(f"comment_{method}", "").strip()  # Comment is optional

            ratings[f"interpretability_{method}"] = interpretability
            ratings[f"plausibility_{method}"] = plausibility
            ratings[f"visual_coherence_{method}"] = visual_coherence
            ratings[f"comment_{method}"] = comment

            if not interpretability:
                missing_fields.append(f"Interpretability ({method})")
            if not plausibility:
                missing_fields.append(f"Plausibility ({method})")
            if not visual_coherence:
                missing_fields.append(f"Visual Coherence ({method})")

        if not original_image:
            missing_fields.append("Original Image")

        if missing_fields:
            flash("Missing fields: " + ", ".join(missing_fields), "error")
            # Re-use the same image if possible
            selected_image = next((item for item in image_data if item["original"] == original_image), None)
            if not selected_image:
                selected_image = random.choice(image_data)
            return render_template("index.html", image_data=selected_image, timestamp=int(time.time()))
        
        # Record the current time and prepare the CSV row.
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        row = [current_time, original_image]
        for method in methods:
            row.append(ratings[f"interpretability_{method}"])
            row.append(ratings[f"plausibility_{method}"])
            row.append(ratings[f"visual_coherence_{method}"])
            row.append(ratings[f"comment_{method}"])
        
        # Save the evaluation row in the CSV.
        with open(responses_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        
        return redirect(url_for("thank_you"))
    
    # GET request: select a random image set for evaluation.
    selected_image = random.choice(image_data)
    return render_template("index.html", image_data=selected_image, timestamp=int(time.time()))

@app.route("/thank_you")
def thank_you():
    return render_template("thank_you.html")

# (Optional) Admin view remains available.
@app.route("/admin/responses")
def admin_responses():
    responses = []
    if os.path.exists(responses_file):
        with open(responses_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                responses.append(row)
    return render_template("admin_responses.html", responses=responses)

if __name__ == "__main__":
    app.run(debug=True)
