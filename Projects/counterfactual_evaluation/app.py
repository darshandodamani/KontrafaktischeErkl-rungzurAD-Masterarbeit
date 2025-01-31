from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import random

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define the list of image pairs (Original - Counterfactual)
image_pairs = [
    {"original": "static/images/original1.png", "counterfactual": "static/images/cf1.png", 
     "original_label": "STOP", "cf_label": "GO"},
    
    {"original": "static/images/original2.png", "counterfactual": "static/images/cf2.png", 
     "original_label": "GO", "cf_label": "STOP"},
    
    {"original": "static/images/original3.png", "counterfactual": "static/images/cf3.png", 
     "original_label": "STOP", "cf_label": "GO"},
    
    {"original": "static/images/original4.png", "counterfactual": "static/images/cf4.png", 
     "original_label": "GO", "cf_label": "STOP"},
]

# Predefined classifier outputs for each image pair
classifier_outputs = {
    0: {"original": {"class": "stop", "confidence": 0.95},
        "counterfactual": {"class": "go", "confidence": 0.80}},

    1: {"original": {"class": "go", "confidence": 0.85},
        "counterfactual": {"class": "stop", "confidence": 0.75}},

    2: {"original": {"class": "stop", "confidence": 0.92},
        "counterfactual": {"class": "go", "confidence": 0.70}},

    3: {"original": {"class": "go", "confidence": 0.88},
        "counterfactual": {"class": "stop", "confidence": 0.65}},
}

csv_file = "responses.csv"
# columns = [
#     "Participant ID", "Image Pair", "Interpretability", "Plausibility", 
#     "Actionability", "Trust in AI", "Visual Coherence", "Classifier Validity",
#     "Classifier Plausibility", "Classifier Feature Attribution", "Comments"
# ]
columns = [
    "Participant ID", "Image Pair", "Interpretability", "Plausibility", 
    "Actionability", "Trust in AI", "Visual Coherence", "Comments"
]

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# def evaluate_classifier(image_pair_idx):
#     """Retrieve predefined classifier outputs."""
#     original_pred = classifier_outputs[image_pair_idx]["original"]
#     cf_pred = classifier_outputs[image_pair_idx]["counterfactual"]

#     classifier_validity = 1 if original_pred["class"] != cf_pred["class"] else 0
#     classifier_plausibility = int(cf_pred["confidence"] * 5)
#     confidence_drop = original_pred["confidence"] - cf_pred["confidence"]
#     classifier_feature = int((confidence_drop / 0.4) * 5)

#     return classifier_validity, classifier_plausibility, classifier_feature

@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        session["participant_id"] = request.form.get("participant_id")
        session["image_index"] = 0  # Reset image index when a new participant starts
        return redirect(url_for("index"))
    return render_template("welcome.html")

@app.route("/evaluate", methods=["GET", "POST"])
def index():
    """Show new images each time a participant fills the form again."""
    if "participant_id" not in session:
        return redirect(url_for("welcome"))

    # Ensure session image index exists
    if "image_index" not in session:
        session["image_index"] = 0

    if request.method == "POST":
        image_pair_idx = session["image_index"]
        
        # Get human ratings
        response_data = {
            "Participant ID": session["participant_id"],
            "Image Pair": image_pair_idx,
            "Interpretability": request.form.get("interpretability"),
            "Plausibility": request.form.get("plausibility"),
            "Actionability": request.form.get("actionability"),
            "Trust in AI": request.form.get("trust"),
            "Visual Coherence": request.form.get("coherence"),
            "Comments": request.form.get("comments") or "No comments"
        }

        # # Retrieve classifier evaluations automatically
        # classifier_validity, classifier_plausibility, classifier_feature = evaluate_classifier(image_pair_idx)
        # response_data["Classifier Validity"] = classifier_validity
        # response_data["Classifier Plausibility"] = classifier_plausibility
        # response_data["Classifier Feature Attribution"] = classifier_feature

        # Save to CSV
        new_data = pd.DataFrame([response_data])
        new_data.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

        # Change to next image
        session["image_index"] = (session["image_index"] + 1) % len(image_pairs)

        return redirect(url_for("thank_you"))

    # Select current image pair
    image_pair_idx = session["image_index"]
    images = image_pairs[image_pair_idx]

    return render_template("index.html", images=images, image_pair_idx=image_pair_idx)

@app.route("/thank_you")
def thank_you():
    """Thank you page after form submission."""
    return render_template("thank_you.html")

if __name__ == "__main__":
    app.run(debug=True)
