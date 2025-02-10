#location: Projects/result_calculate/masking_performance_report.py
"""
masking_performance_report.py

This script aggregates, compares, and summarizes the performance of various masking-based 
counterfactual explanation methods. It processes results from multiple masking techniques, 
extracts key evaluation metrics, and generates a comparative performance report.

"""

import os
import pandas as pd
import logging

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Configuration: Paths to Result CSV Files
# ------------------------------------------------------------------------------
RESULTS_DIR = "results/method_comparision"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths to individual summary CSV files
SUMMARY_CSV_FILES = {
    "Grid-Based Masking": "results/masking/grid_based/grid_based_summary.csv",
    "Object Detection Masking": "results/masking/object_detection/object_detection_summary.csv",
    "LIME on Images": "results/masking/lime_on_images/lime_on_image_summary.csv",
    "LIME on Latent Features": "results/masking/lime_on_latent/lime_on_latent_summary.csv",
}

# Output file for comparative summary
SUMMARY_OUTPUT_CSV = os.path.join(RESULTS_DIR, "comparative_summary_across_methods_with_percentage.csv")

# ------------------------------------------------------------------------------
# Function: Load Data Safely
# ------------------------------------------------------------------------------
def load_csv_safe(filepath):
    """Loads a CSV file safely, ensuring errors are handled gracefully."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None

# ------------------------------------------------------------------------------
# Function: Extract Key Metrics
# ------------------------------------------------------------------------------
def extract_metrics(data, method_name):
    """Extracts key performance metrics from a given summary DataFrame."""
    try:
        total_entries = int(data.loc[data["Metrics"] == "Total", "Total Count"].values[0])
        total_go_cases = int(data.loc[data["Metrics"] == "Total GO", "Total Count"].values[0])
        total_stop_cases = int(data.loc[data["Metrics"] == "Total STOP", "Total Count"].values[0])

        return {
            "Method": method_name,
            "GO (Counterfactual Found)": f'{data.loc[data["Metrics"] == "GO (Counterfactual Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "GO (Counterfactual Found)", "Percentage"].values[0]})',
            "STOP (Counterfactual Found)": f'{data.loc[data["Metrics"] == "STOP (Counterfactual Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "STOP (Counterfactual Found)", "Percentage"].values[0]})',
            "GO (No Counterfactual)": f'{data.loc[data["Metrics"] == "GO (Counterfactual Not Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "GO (Counterfactual Not Found)", "Percentage"].values[0]})',
            "STOP (No Counterfactual)": f'{data.loc[data["Metrics"] == "STOP (Counterfactual Not Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "STOP (Counterfactual Not Found)", "Percentage"].values[0]})',
            "Total Time Taken (s)": float(data.loc[data["Metrics"] == "Total Time Taken", "Count"].values[0]),
            "Total Entries": total_entries,
            "Total GO Cases": total_go_cases,
            "Total STOP Cases": total_stop_cases,
        }
    except Exception as e:
        logging.error(f"Error extracting metrics for {method_name}: {e}")
        return None

# ------------------------------------------------------------------------------
# Process and Aggregate Metrics
# ------------------------------------------------------------------------------
summary_list = []
for method, csv_path in SUMMARY_CSV_FILES.items():
    df = load_csv_safe(csv_path)
    if df is not None:
        metrics = extract_metrics(df, method)
        if metrics:
            summary_list.append(metrics)

# Convert to DataFrame
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    
    # Save the comparative summary to a CSV file
    summary_df.to_csv(SUMMARY_OUTPUT_CSV, index=False)
    logging.info(f"Comparative summary saved to {SUMMARY_OUTPUT_CSV}")

    # Print results
    print("\nComparative Summary Across Methods:")
    print(summary_df.to_string(index=False))
else:
    logging.error("No valid summary data available. Exiting.")
