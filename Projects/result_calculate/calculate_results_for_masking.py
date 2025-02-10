#location: Projects/result_calculate/calculate_results_for_masking.py

"""
This script processes and summarizes the results of different counterfactual masking methods.
"""
import os
import sys
import csv
import time
import logging
from typing import List, Dict

import pandas as pd

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Configuration and File Paths
# ------------------------------------------------------------------------------
# Input CSV file paths for each masking method
methods_results: Dict[str, str] = {
    "grid_based": "results/masking/grid_based_masking_results.csv",
    "lime_on_image": "results/masking/lime_on_image_masking_results.csv",
    "object_detection": "results/masking/object_detection_masking_results.csv",
    "lime_on_latent": "results/masking/lime_on_latent_masking_results.csv"
}

# Output summary CSV file paths for each method
output_summaries: Dict[str, str] = {
    "grid_based": os.path.join("results", "masking", "grid_based", "grid_based_summary.csv"),
    "lime_on_image": os.path.join("results", "masking", "lime_on_images", "lime_on_image_summary.csv"),
    "object_detection": os.path.join("results", "masking", "object_detection", "object_detection_summary.csv"),
    "lime_on_latent": os.path.join("results", "masking", "lime_on_latent", "lime_on_latent_summary.csv")
}

# Expected number of fields per row (based on the header in each CSV)
EXPECTED_FIELDS = 14

# ------------------------------------------------------------------------------
# Helper Function: find_bad_rows()
# ------------------------------------------------------------------------------
def find_bad_rows(csv_file: str, expected_fields: int) -> None:
    """
    Scans the CSV file and logs/prints any row whose number of fields does not match
    the expected number.
    
    Args:
        csv_file (str): Path to the CSV file.
        expected_fields (int): Expected number of fields per row.
    """
    bad_rows: List[tuple] = []
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # read header (line 1)
        line_num = 1
        for row in reader:
            line_num += 1
            if len(row) != expected_fields:
                bad_rows.append((line_num, row))
    if bad_rows:
        logging.warning("Found the following row(s) with unexpected number of fields:")
        for line_num, row in bad_rows:
            logging.warning(f"Line {line_num}: Expected {expected_fields} fields, got {len(row)}: {row}")
            print(f"Line {line_num}: {row}")
    else:
        logging.info("No formatting issues found in the CSV file.")

# ------------------------------------------------------------------------------
# Function: compute_summary()
# ------------------------------------------------------------------------------
def compute_summary(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Computes summary metrics from a masking results CSV and saves the summary.
    
    For each prediction class ("GO" and "STOP"), this function calculates:
      - Total count,
      - Count and percentage of counterfactual found and not found.
    It also computes overall totals and total time taken.
    
    First, it counts the expected number of data rows (excluding the header)
    and then reads the CSV using the Python engine with on_bad_lines="skip".
    If the row count does not match the expectation, it calls find_bad_rows().
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path where the summary CSV will be saved.
    
    Returns:
        pd.DataFrame: DataFrame containing the summary metrics.
    """
    # Count expected data rows (excluding header)
    try:
        with open(input_csv, "r", newline="") as f:
            expected_rows = sum(1 for _ in f) - 1
        logging.info(f"Expected data rows (excluding header): {expected_rows}")
    except Exception as e:
        logging.error(f"Error counting lines in {input_csv}: {e}")
        sys.exit(1)
    
    # Read the CSV file with bad lines skipped
    try:
        data = pd.read_csv(input_csv, engine="python", on_bad_lines="skip")
        read_rows = data.shape[0]
        logging.info(f"Read {read_rows} data rows from {input_csv}")
        if read_rows != expected_rows:
            logging.warning(
                f"Discrepancy detected: Expected {expected_rows} rows, but read {read_rows} rows. "
                "Some rows may have been skipped due to formatting issues."
            )
            find_bad_rows(input_csv, expected_fields=EXPECTED_FIELDS)
    except Exception as e:
        logging.error(f"Error reading input CSV file {input_csv}: {e}")
        sys.exit(1)
    
    metrics: List[Dict[str, str]] = []
    
    # Overall totals
    total_time = data["Time Taken (s)"].sum()
    total_entries = len(data)
    
    overall_ce_found = 0
    overall_ce_not_found = 0
    
    # Process each prediction class ("GO" and "STOP")
    for prediction_class in ["GO", "STOP"]:
        class_data = data[data["Prediction (Before Masking)"] == prediction_class]
        total_cases = len(class_data)
        
        ce_found_count = len(class_data[class_data["Counterfactual Found"] == True])
        ce_not_found_count = len(class_data[class_data["Counterfactual Found"] == False])
        
        ce_found_percentage = (ce_found_count / total_cases * 100) if total_cases > 0 else 0
        ce_not_found_percentage = (ce_not_found_count / total_cases * 100) if total_cases > 0 else 0
        
        overall_ce_found += ce_found_count
        overall_ce_not_found += ce_not_found_count
        
        metrics.append({
            "Metrics": f"Total {prediction_class}",
            "Total Count": str(total_cases),
            "Count": "",
            "Percentage": ""
        })
        metrics.append({
            "Metrics": f"{prediction_class} (Counterfactual Found)",
            "Total Count": "",
            "Count": str(ce_found_count),
            "Percentage": f"{ce_found_percentage:.2f}%"
        })
        metrics.append({
            "Metrics": f"{prediction_class} (Counterfactual Not Found)",
            "Total Count": "",
            "Count": str(ce_not_found_count),
            "Percentage": f"{ce_not_found_percentage:.2f}%"
        })
    
    overall_found_percentage = (overall_ce_found / total_entries * 100) if total_entries > 0 else 0
    overall_not_found_percentage = (overall_ce_not_found / total_entries * 100) if total_entries > 0 else 0
    
    metrics.append({
        "Metrics": "Total",
        "Total Count": str(total_entries),
        "Count": "",
        "Percentage": ""
    })
    metrics.append({
        "Metrics": "Total (Counterfactual Found)",
        "Total Count": "",
        "Count": str(overall_ce_found),
        "Percentage": f"{overall_found_percentage:.2f}%"
    })
    metrics.append({
        "Metrics": "Total (Counterfactual Not Found)",
        "Total Count": "",
        "Count": str(overall_ce_not_found),
        "Percentage": f"{overall_not_found_percentage:.2f}%"
    })
    metrics.append({
        "Metrics": "Total Time Taken",
        "Total Count": "",
        "Count": f"{total_time:.2f}",
        "Percentage": ""
    })
    
    summary_table = pd.DataFrame(metrics)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    summary_table.to_csv(output_csv, index=False)
    return summary_table

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function that computes and saves summary metrics for all masking methods.
    
    For each method defined in the methods_results dictionary, this function:
      - Computes summary metrics using compute_summary(),
      - Saves the summary to the designated output CSV file,
      - Prints the summary and the absolute path of the saved summary to the terminal.
    """
    for method, input_csv in methods_results.items():
        output_csv = output_summaries[method]
        logging.info(f"Processing {method} results from {input_csv}")
        summary_table = compute_summary(input_csv, output_csv)
        
        print(f"\n{method.upper()} Masking Results Summary:")
        print(summary_table)
        
        abs_output_csv = os.path.abspath(output_csv)
        logging.info(f"Summary for {method} saved to {abs_output_csv}")
        print(f"Summary for {method} saved to {abs_output_csv}")

if __name__ == "__main__":
    main()
