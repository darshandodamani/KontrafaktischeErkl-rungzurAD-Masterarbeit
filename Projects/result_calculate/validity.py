#location: Projects/result_calculate/validity.py
import os
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration and File Paths
# ------------------------------------------------------------------------------
# Define the paths for different methods' result files
METHODS_FILES = {
    "Grid-Based Masking": "results/masking/grid_based_masking_results.csv",
    "Object Detection": "results/masking/object_detection_masking_results.csv",
    "LIME on Images": "results/masking/lime_on_image_masking_results.csv",
    "LIME on Latent Features": "results/masking/lime_on_latent_masking_results.csv"
}

# Output directory for validity check
VALIDITY_DIR = "results/validity_check"
os.makedirs(VALIDITY_DIR, exist_ok=True)

# Output CSV file path
VALIDITY_CSV_PATH = os.path.join(VALIDITY_DIR, "validity_summary.csv")

# ------------------------------------------------------------------------------
# Function: Compute Validity
# ------------------------------------------------------------------------------
def calculate_validity(df: pd.DataFrame, method_name: str) -> dict:
    """
    Computes validity percentage for a given method by checking the number
    of successfully found counterfactuals.
    
    Args:
        df (pd.DataFrame): The dataset containing counterfactual results.
        method_name (str): The name of the method.
    
    Returns:
        dict: A dictionary containing method validity statistics.
    """
    total_counterfactuals = len(df)
    successful_counterfactuals = df[df["Counterfactual Found"] == True].shape[0]
    validity = (successful_counterfactuals / total_counterfactuals * 100) if total_counterfactuals > 0 else 0

    return {
        "Method": method_name,
        "Total Counterfactuals": total_counterfactuals,
        "Successful Counterfactuals": successful_counterfactuals,
        "Validity (%)": round(validity, 2)
    }

# ------------------------------------------------------------------------------
# Load Data and Compute Validity
# ------------------------------------------------------------------------------
validity_results = []

for method, filepath in METHODS_FILES.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        validity_results.append(calculate_validity(df, method))
    else:
        print(f"⚠ Warning: {filepath} not found. Skipping {method}.")

# Convert results into a DataFrame
validity_df = pd.DataFrame(validity_results)

# Save the validity summary as a CSV file
validity_df.to_csv(VALIDITY_CSV_PATH, index=False)

# ------------------------------------------------------------------------------
# Print Validity Summary
# ------------------------------------------------------------------------------
print("\n📊 Validity Summary Across Methods:")
print(validity_df.to_string(index=False))
print(f"\n✅ Validity summary saved to: {VALIDITY_CSV_PATH}")
