#location: Projects/result_calculate/masking_methods_comparison.py
"""
This script compares and analyzes the performance of multiple masking-based counterfactual 
explanation methods. It processes the results from different techniques, extracts key statistics, 
generates a **comparative summary table**, and visualizes the overlaps using a **Venn diagram**.

"""

import os
import sys
import logging
from typing import Set, Dict

import pandas as pd
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

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
# Input CSV files for each masking method
GRID_CSV = "results/masking/grid_based_masking_results.csv"
OBJECT_DETECTION_CSV = "results/masking/object_detection_masking_results.csv"
LIME_LATENT_CSV = "results/masking/lime_on_latent_masking_results.csv"
LIME_IMAGE_CSV = "results/masking/lime_on_image_masking_results.csv"

# Output paths for generated results
METHOD_COMPARISON_DIR = os.path.join("results", "method_comparision")
os.makedirs(METHOD_COMPARISON_DIR, exist_ok=True)
TABLE_OUTPUT_PATH = os.path.join(METHOD_COMPARISON_DIR, "method_comparison_table.csv")

VENN_DIAGRAM_DIR = os.path.join("plots", "venn_diagram")
os.makedirs(VENN_DIAGRAM_DIR, exist_ok=True)
VENN_DIAGRAM_PATH = os.path.join(VENN_DIAGRAM_DIR, "venn_4.png")

BAR_CHART_PATH = os.path.join(VENN_DIAGRAM_DIR, "bar_chart_explanations.png")  # New bar chart path

# ------------------------------------------------------------------------------
# Data Loading Functions
# ------------------------------------------------------------------------------
def load_dataframe(csv_file: str) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {csv_file} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {e}")
        sys.exit(1)

def get_counterfactual_set(df: pd.DataFrame) -> Set[str]:
    """
    Extracts the set of image filenames where 'Counterfactual Found' is True.

    Args:
        df (pd.DataFrame): DataFrame containing masking results.

    Returns:
        Set[str]: Set of image filenames with counterfactual found.
    """
    return set(df[df["Counterfactual Found"] == True]["Image File"])

# ------------------------------------------------------------------------------
# Generate Comparison Table
# ------------------------------------------------------------------------------
def generate_comparison_table(all_images: Set[str],
                              grid_ce: Set[str],
                              lime_latent_ce: Set[str],
                              lime_image_ce: Set[str],
                              object_detection_ce: Set[str]) -> pd.DataFrame:
    """
    Generates a comparison table indicating which images are explained by each method.

    Args:
        all_images (Set[str]): Set of all image filenames.
        grid_ce, lime_latent_ce, lime_image_ce, object_detection_ce (Set[str]):
            Sets of image filenames with counterfactuals found for each method.

    Returns:
        pd.DataFrame: Comparison table as a DataFrame.
    """
    table = []
    for image in sorted(all_images):
        table.append({
            "Image File": image,
            "Grid": "True" if image in grid_ce else "False",
            "LIME on Latent Features": "True" if image in lime_latent_ce else "False",
            "LIME on Images": "True" if image in lime_image_ce else "False",
            "Object Detection": "True" if image in object_detection_ce else "False",
        })
    return pd.DataFrame(table)

# ------------------------------------------------------------------------------
# Generate Venn Diagram
# ------------------------------------------------------------------------------
def generate_venn_diagram(sets: Dict[str, Set[str]]) -> None:
    """
    Generates and saves a Venn diagram for the provided sets using venny4py.

    Args:
        sets (Dict[str, Set[str]]): Dictionary mapping method names to sets of image filenames.
    """
    plt.figure(figsize=(12, 8))
    venny4py(sets=sets)
    plt.title("Venn Diagram of Counterfactual Explanation Coverage", fontsize=16)
    plt.savefig(VENN_DIAGRAM_PATH, bbox_inches='tight')
    plt.close()
    logging.info(f"Venn diagram saved as '{os.path.abspath(VENN_DIAGRAM_PATH)}'")

# ------------------------------------------------------------------------------
# Generate Bar Chart with Labels
# ------------------------------------------------------------------------------
def generate_bar_chart(grid_ce: Set[str], 
                       lime_latent_ce: Set[str], 
                       lime_image_ce: Set[str], 
                       object_detection_ce: Set[str]) -> None:
    """
    Generates and saves a bar chart showing the number of images explained by each method.
    Labels are added above each bar for clarity.

    Args:
        grid_ce, lime_latent_ce, lime_image_ce, object_detection_ce (Set[str]):
            Sets of image filenames with counterfactuals found for each method.
    """
    methods = ["Grid-Based", "LIME on Latent", "LIME on Images", "Object Detection"]
    counts = [len(grid_ce), len(lime_latent_ce), len(lime_image_ce), len(object_detection_ce)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, counts, color=['blue', 'orange', 'green', 'red'])

    # Add numerical labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, str(yval), ha='center', fontsize=12, fontweight='bold')

    plt.xlabel("Masking Methods", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Images Explained", fontsize=12, fontweight='bold')
    plt.title("Counterfactual Explanations per Method", fontsize=14, fontweight='bold')
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the chart
    plt.savefig(BAR_CHART_PATH, bbox_inches='tight')
    plt.close()

    logging.info(f"Bar chart saved as '{os.path.abspath(BAR_CHART_PATH)}'")


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function to generate the method comparison table, Venn diagram, 
    bar chart, and summary statistics.
    """
    # Load DataFrames for each method
    grid_df = load_dataframe(GRID_CSV)
    object_detection_df = load_dataframe(OBJECT_DETECTION_CSV)
    lime_latent_df = load_dataframe(LIME_LATENT_CSV)
    lime_image_df = load_dataframe(LIME_IMAGE_CSV)

    # Create sets of images with counterfactual found for each method
    grid_ce = get_counterfactual_set(grid_df)
    object_detection_ce = get_counterfactual_set(object_detection_df)
    lime_latent_ce = get_counterfactual_set(lime_latent_df)
    lime_image_ce = get_counterfactual_set(lime_image_df)

    # Combine all unique image files from all methods
    all_images = (set(grid_df["Image File"]) |
                  set(object_detection_df["Image File"]) |
                  set(lime_latent_df["Image File"]) |
                  set(lime_image_df["Image File"]))

    # Generate comparison table and save it
    table_df = generate_comparison_table(all_images, grid_ce, lime_latent_ce, lime_image_ce, object_detection_ce)
    table_df.to_csv(TABLE_OUTPUT_PATH, index=False)
    logging.info(f"Comparison table saved as '{os.path.abspath(TABLE_OUTPUT_PATH)}'")

    # Generate Venn diagram
    venn_sets = {
        "Grid-Based Masking": grid_ce,
        "LIME on Latent Features": lime_latent_ce,
        "LIME on Images": lime_image_ce,
        "Object Detection-Based": object_detection_ce,
    }
    generate_venn_diagram(venn_sets)

    # Generate Bar Chart
    generate_bar_chart(grid_ce, lime_latent_ce, lime_image_ce, object_detection_ce)

    print("\nAll visualizations and reports have been successfully generated.")

if __name__ == "__main__":
    main()
