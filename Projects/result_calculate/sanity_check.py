#location: Projects/result_calculate/sanity_check.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Setup: Define Paths & Load Data
# ------------------------------------------------------------------------------
# File paths for different methods
METHODS_FILES = {
    "Grid-Based Masking": "results/masking/grid_based_masking_results.csv",
    "Object Detection": "results/masking/object_detection_masking_results.csv",
    "LIME on Images": "results/masking/lime_on_image_masking_results.csv",
    "LIME on Latent Features": "results/masking/lime_on_latent_masking_results.csv"
}

# Ensure directories exist
PLOTS_DIR = "plots/sanity_check"
RESULTS_DIR = "results/sanity_check"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------
df_list = []
for method, filepath in METHODS_FILES.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["Method"] = method  # Add method column
        df_list.append(df)
    else:
        print(f"⚠ Warning: {filepath} not found. Skipping...")

# Merge all data
df = pd.concat(df_list, ignore_index=True)

# ------------------------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------------------------
# Select relevant columns
df = df[["Method", "Image File", "Counterfactual Found", "PSNR", "SSIM", "MSE", "UQI", "VIFP"]]

# Filter **only rows where Counterfactual Found == True**
df_true = df[df["Counterfactual Found"] == True]  # ✅ This removes "False" rows

# ------------------------------------------------------------------------------
# Function: Save Boxplots
# ------------------------------------------------------------------------------
def save_boxplot(metric):
    """
    Generates and saves boxplots for a given metric comparing methods.

    Args:
        metric (str): The metric to plot (e.g., "PSNR", "SSIM").
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_true, x="Counterfactual Found", y=metric, hue="Method")
    plt.title(f"{metric} vs. Counterfactual Found", fontsize=14)
    plt.xlabel("Counterfactual Found", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(title="Method", fontsize=10)
    plot_path = os.path.join(PLOTS_DIR, f"{metric.lower()}_boxplot.png")
    plt.savefig(plot_path)
    plt.close()  # Close figure to prevent display errors
    print(f"📊 {metric} boxplot saved at: {plot_path}")

# Generate and save plots for all metrics
for metric in ["PSNR", "SSIM", "MSE", "UQI", "VIFP"]:
    save_boxplot(metric)

# ------------------------------------------------------------------------------
# Compute Summary Statistics (Only for True Cases)
# ------------------------------------------------------------------------------
summary = df_true.groupby(["Method"]).agg({
    "PSNR": ["mean", "std", "min", "max"],
    "SSIM": ["mean", "std", "min", "max"],
    "MSE": ["mean", "std", "min", "max"],
    "UQI": ["mean", "std", "min", "max"],
    "VIFP": ["mean", "std", "min", "max"]
}).reset_index()

# Flatten MultiIndex Columns
summary.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]

# ------------------------------------------------------------------------------
# Save Summary as CSV & Excel
# ------------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "sanity_check_image_vs_ce_metrics.csv")

summary.to_csv(csv_path, index=False)

print(f"✅ Sanity check summary saved to:\n📄 CSV: {csv_path}")
