import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths to train and test CSV files for all methods
methods_files = {
    "Grid-Based": ("plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv", 
                   "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"),
    "Object Detection": ("plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv",
                         "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"),
    "LIME on Images": ("plots/lime_on_images/lime_on_image_masking_train_results.csv",
                       "plots/lime_on_images/lime_on_image_masking_test_results.csv"),
    "LIME on Latent": ("plots/lime_plots/lime_latent_masking_train_results_updated.csv",
                       "plots/lime_plots/lime_latent_masking_test_results_updated.csv")
}

# Metrics to analyze
metrics = ["SSIM", "MSE", "PSNR", "UQI", "VIFP"]

# Prepare to collect extreme values
extreme_values = {
    "Metric": [],
    "Method": [],
    "Dataset": [],
    "Image File": [],
    "Value": [],
    "Type": []
}

# Function to process train and test datasets for a method
def process_method(method_name, train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    for dataset_name, data in [("Train", train_data), ("Test", test_data), ("Combined", combined_data)]:
        for metric in metrics:
            # Ignore rows with NaN
            valid_data = data[~data[metric].isna()]

            if valid_data.empty:
                continue

            # Find max and min
            max_val = valid_data[metric].max()
            min_val = valid_data[metric].min()

            # Append largest value
            extreme_values["Metric"].append(metric)
            extreme_values["Method"].append(method_name)
            extreme_values["Dataset"].append(dataset_name)
            extreme_values["Image File"].append(valid_data.loc[valid_data[metric].idxmax(), "Image File"])
            extreme_values["Value"].append(max_val)
            extreme_values["Type"].append("Largest")

            # Append smallest value
            extreme_values["Metric"].append(metric)
            extreme_values["Method"].append(method_name)
            extreme_values["Dataset"].append(dataset_name)
            extreme_values["Image File"].append(valid_data.loc[valid_data[metric].idxmin(), "Image File"])
            extreme_values["Value"].append(min_val)
            extreme_values["Type"].append("Smallest")

    return combined_data

# Collect all combined data for visualization
all_combined_data = []

# Process each method
for method_name, (train_path, test_path) in methods_files.items():
    combined_data = process_method(method_name, train_path, test_path)
    all_combined_data.append(combined_data)

# Combine all data across methods
all_combined_data = pd.concat(all_combined_data, ignore_index=True)

# Convert extreme values to DataFrame
extreme_values_df = pd.DataFrame(extreme_values)

# Save extreme values to CSV
extreme_values_df.to_csv("plots/similarity_metrics_extremes.csv", index=False)

# Summary of extreme values
print("\nSummary of Extreme Similarity Metrics:")
print(extreme_values_df)

# Visualize distributions for each metric
plt.figure(figsize=(12, 8))
for metric in metrics:
    plt.hist(all_combined_data[metric].dropna(), bins=30, alpha=0.6, label=metric)
plt.xlabel("Metric Values")
plt.ylabel("Frequency")
plt.title("Distribution of Similarity Metrics Across Train and Test Data")
plt.legend()
plt.savefig("plots/similarity_metrics_distribution.png")
