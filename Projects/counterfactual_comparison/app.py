import csv
import random
import os
import time

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- FastAPI Setup ---
app = FastAPI()

# Set the base directory (directory of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the CSV file for logging responses (for evaluations)
responses_file = os.path.join(BASE_DIR, "responses.csv")

# Ensure the responses CSV file exists with headers (anonymous columns)
def initialize_responses_csv():
    if not os.path.exists(responses_file) or os.stat(responses_file).st_size == 0:
        with open(responses_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Image",
                "Counterfactual_1_Interpretability", "Counterfactual_1_Plausibility", "Counterfactual_1_VisualCoherence", "Counterfactual_1_Comments",
                "Counterfactual_2_Interpretability", "Counterfactual_2_Plausibility", "Counterfactual_2_VisualCoherence", "Counterfactual_2_Comments",
                "Counterfactual_3_Interpretability", "Counterfactual_3_Plausibility", "Counterfactual_3_VisualCoherence", "Counterfactual_3_Comments"
            ])

initialize_responses_csv()

# --- Load Label Data from CSV ---
# Expected CSV header: "Image File", "Prediction (Before Masking)", "Prediction (After Masking)"
def load_labels():
    labels = {}
    # CSV file is in static/ directory.
    label_csv = os.path.join(BASE_DIR, "static", "label_data.csv")
    if os.path.exists(label_csv):
        with open(label_csv, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row.get("Image File")
                if filename:
                    labels[filename] = row
    return labels

labels_data = load_labels()

# --- Get Random Image Data ---
def get_random_image_data():
    original_dir = os.path.join(BASE_DIR, "static", "images", "original")
    all_files = os.listdir(original_dir)
    # Filter common image file types
    image_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        raise Exception("No images found in the original directory.")
    chosen = random.choice(image_files)
    data = {
        "original": "images/original/" + chosen,
        "method1": "images/method1/" + chosen,
        "method2": "images/method2/" + chosen,
        "method3": "images/method3/" + chosen,
        "labels": {}
    }
    if chosen in labels_data:
        row = labels_data[chosen]
        data["labels"]["original"] = row.get("Prediction (Before Masking)", "N/A")
        # All methods use the same "After" prediction
        after_label = row.get("Prediction (After Masking)", "N/A")
        data["labels"]["method1"] = after_label
        data["labels"]["method2"] = after_label
        data["labels"]["method3"] = after_label
    else:
        data["labels"]["original"] = "N/A"
        data["labels"]["method1"] = "N/A"
        data["labels"]["method2"] = "N/A"
        data["labels"]["method3"] = "N/A"
    return data

# --- Mount Static Files and Setup Templates ---
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Web Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

@app.get("/evaluate", response_class=HTMLResponse, name="evaluate")
async def evaluate_get(request: Request):
    selected_image = get_random_image_data()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_data": selected_image, "timestamp": int(time.time())}
    )

@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate_post(
    request: Request,
    image: str = Form(...),
    # Ratings for counterfactual image 1 (method1)
    interpretability_method1: str = Form(...),
    plausibility_method1: str = Form(...),
    visual_coherence_method1: str = Form(...),
    comments_method1: str = Form(""),
    # Ratings for counterfactual image 2 (method2)
    interpretability_method2: str = Form(...),
    plausibility_method2: str = Form(...),
    visual_coherence_method2: str = Form(...),
    comments_method2: str = Form(""),
    # Ratings for counterfactual image 3 (method3)
    interpretability_method3: str = Form(...),
    plausibility_method3: str = Form(...),
    visual_coherence_method3: str = Form(...),
    comments_method3: str = Form("")
):
    # Save only the filename (e.g., "town7_009556.png")
    image_filename = os.path.basename(image)
    form_data = {
        "image": image_filename,
        "Counterfactual_1": {
            "interpretability": interpretability_method1,
            "plausibility": plausibility_method1,
            "visual_coherence": visual_coherence_method1,
            "comments": comments_method1,
        },
        "Counterfactual_2": {
            "interpretability": interpretability_method2,
            "plausibility": plausibility_method2,
            "visual_coherence": visual_coherence_method2,
            "comments": comments_method2,
        },
        "Counterfactual_3": {
            "interpretability": interpretability_method3,
            "plausibility": plausibility_method3,
            "visual_coherence": visual_coherence_method3,
            "comments": comments_method3,
        }
    }
    print("Form Data Received:", form_data)
    with open(responses_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            image_filename,
            interpretability_method1, plausibility_method1, visual_coherence_method1, comments_method1,
            interpretability_method2, plausibility_method2, visual_coherence_method2, comments_method2,
            interpretability_method3, plausibility_method3, visual_coherence_method3, comments_method3
        ])
    return RedirectResponse(url="/thank_you", status_code=302)

@app.get("/thank_you", response_class=HTMLResponse)
async def thank_you(request: Request):
    return templates.TemplateResponse("thank_you.html", {"request": request})
