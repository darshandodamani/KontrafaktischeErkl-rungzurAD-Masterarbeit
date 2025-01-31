import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")

# File paths for different methods
grid_csv = "plots/grid_based_masking_results.csv"
object_detection_csv = "plots/object_detection_masking_results.csv"
lime_on_images_csv = "plots/lime_on_image_masking_results.csv"
lime_on_latent_csv = "plots/lime_on_latent_masking_results.csv"

# Load CSVs
grid_df = pd.read_csv(grid_csv)
object_detection_df = pd.read_csv(object_detection_csv)
lime_image_df = pd.read_csv(lime_on_images_csv)
lime_latent_df = pd.read_csv(lime_on_latent_csv)

# Add method names
grid_df["Method"] = "Grid-Based Masking"
object_detection_df["Method"] = "Object Detection"
lime_image_df["Method"] = "LIME on Images"
lime_latent_df["Method"] = "LIME on Latent Features"

# Combine all data
df = pd.concat([grid_df, object_detection_df, lime_image_df, lime_latent_df], ignore_index=True)

# Select relevant columns
df = df[["Method", "Image File", "Counterfactual Found", "PSNR", "SSIM", "MSE", "UQI", "VIFP"]]
df["Counterfactual Found"] = df["Counterfactual Found"].astype(str)  # Convert for visualization

# 🔹 Function to Save Boxplots
def save_boxplot(metric):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Counterfactual Found", y=metric, hue="Method")
    plt.title(f"{metric} vs. Counterfactual Found")
    plt.savefig(f"plots/{metric.lower()}_boxplot.png")  # Save instead of show
    plt.close()  # Prevent showing errors

# Generate and save plots
for metric in ["PSNR", "SSIM", "MSE", "UQI", "VIFP"]:
    save_boxplot(metric)

# 🔹 Fix MultiIndex Issue in Excel Export
summary = df.groupby(["Method", "Counterfactual Found"]).agg({
    "PSNR": ["mean", "std", "min", "max"],
    "SSIM": ["mean", "std", "min", "max"],
    "MSE": ["mean", "std", "min", "max"],
    "UQI": ["mean", "std", "min", "max"],
    "VIFP": ["mean", "std", "min", "max"]
}).reset_index()

# Flatten MultiIndex Columns
summary.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]

# Save to Excel
excel_path = "plots/sanity_check_image_vs_ce_metrics.xlsx"
summary.to_excel(excel_path, index=False)

print(f"Sanity check summary saved to: {excel_path}")