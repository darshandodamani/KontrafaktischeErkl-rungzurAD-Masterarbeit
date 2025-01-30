import pandas as pd

# Paths to result CSV files
grid_csv = "plots/grid_based_summary.csv"
object_detection_csv = "plots/object_detection_summary.csv"
lime_on_images_csv = "plots/lime_on_image_summary.csv"
lime_on_latent_csv = "plots/lime_on_latent_summary.csv"

# Load summary CSVs into DataFrames and ensure numeric columns are converted correctly
grid_data = pd.read_csv(grid_csv)
object_detection_data = pd.read_csv(object_detection_csv)
lime_on_images_data = pd.read_csv(lime_on_images_csv)
lime_on_latent_data = pd.read_csv(lime_on_latent_csv)

# Convert Total Count and Count columns to numeric if they exist
for df in [grid_data, object_detection_data, lime_on_images_data, lime_on_latent_data]:
    numeric_columns = ["Total Count", "Count"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

# Function to extract required metrics from the summary DataFrame
def extract_metrics(data, method_name):
    total_entries = data.loc[data["Metrics"] == "Total", "Total Count"].values[0]
    total_go_cases = data.loc[data["Metrics"] == "Total GO", "Total Count"].values[0]
    total_stop_cases = data.loc[data["Metrics"] == "Total STOP", "Total Count"].values[0]
    
    metrics = {
        "Method": method_name,
        "GO (Counterfactual Found)": f'{data.loc[data["Metrics"] == "GO (Counterfactual Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "GO (Counterfactual Found)", "Percentage"].values[0]})',
        "STOP (Counterfactual Found)": f'{data.loc[data["Metrics"] == "STOP (Counterfactual Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "STOP (Counterfactual Found)", "Percentage"].values[0]})',
        "GO (No Counterfactual)": f'{data.loc[data["Metrics"] == "GO (Counterfactual Not Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "GO (Counterfactual Not Found)", "Percentage"].values[0]})',
        "STOP (No Counterfactual)": f'{data.loc[data["Metrics"] == "STOP (Counterfactual Not Found)", "Count"].values[0]} ({data.loc[data["Metrics"] == "STOP (Counterfactual Not Found)", "Percentage"].values[0]})',
        "Total Time Taken (s)": data.loc[data["Metrics"] == "Total Time Taken", "Count"].values[0],
        "Total Entries": total_entries,
        "Total GO Cases": total_go_cases,
        "Total STOP Cases": total_stop_cases,
        # "GO (CF %)": f"{round((data.loc[data['Metrics'] == 'GO (Counterfactual Found)', 'Count'].values[0] / total_go_cases) * 100, 2)}%",
        # "STOP (CF %)": f"{round((data.loc[data['Metrics'] == 'STOP (Counterfactual Found)', 'Count'].values[0] / total_stop_cases) * 100, 2)}%"
    }
    return metrics

# Extract metrics for each method
grid_metrics = extract_metrics(grid_data, "Grid-Based Masking")
object_detection_metrics = extract_metrics(object_detection_data, "Object Detection Masking")
lime_on_images_metrics = extract_metrics(lime_on_images_data, "LIME on Images")
lime_on_latent_metrics = extract_metrics(lime_on_latent_data, "LIME on Latent Features")

# Combine metrics into a single DataFrame
summary = pd.DataFrame([
    grid_metrics,
    object_detection_metrics,
    lime_on_images_metrics,
    lime_on_latent_metrics
])

# Save the comparative summary to a CSV file
summary_csv_path = "plots/comparative_summary_across_methods_with_percentage.csv"
summary.to_csv(summary_csv_path, index=False)

# Print the comparative summary
print("\nComparative Summary Across Methods:")
print(summary)
