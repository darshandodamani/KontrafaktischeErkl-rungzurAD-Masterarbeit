import pandas as pd

# File paths to the individual result CSV files
grid_csv = "Projects/result_calculate/grid_based_results_summary.csv"
object_detection_csv = "Projects/result_calculate/object_detection_results_summary.csv"
lime_latent_csv = "Projects/result_calculate/lime_latent_results_summary.csv"
lime_image_csv = "Projects/result_calculate/lime_image_results_summary.csv"

#original dataset
train_dataset_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
test_dataset_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Method names for labeling
methods = ["Grid-Based Masking", "Object Detection-Based", "LIME-Based Latent Features", "LIME on Image Masking"]

# Load CSV files
grid_df = pd.read_csv(grid_csv)
object_detection_df = pd.read_csv(object_detection_csv)
lime_latent_df = pd.read_csv(lime_latent_csv)
lime_image_df = pd.read_csv(lime_image_csv)

# Load original dataset to calculate total GO and STOP cases
train_data = pd.read_csv(train_dataset_csv)
test_data = pd.read_csv(test_dataset_csv)

# Total GO and STOP counts from the original dataset
total_go = train_data[train_data["label"] == "GO"].shape[0] + test_data[test_data["label"] == "GO"].shape[0]
total_stop = train_data[train_data["label"] == "STOP"].shape[0] + test_data[test_data["label"] == "STOP"].shape[0]

# Helper function to safely retrieve a value
def safe_get_value(df, metric, column="Combined Count", default=0):
    try:
        return df.loc[df["Metric"] == metric, column].values[0]
    except IndexError:
        return default

# # Extract relevant metrics and combine results
# combined_results = []

# for method, df in zip(methods, [grid_df, object_detection_df, lime_latent_df, lime_image_df]):
#     go_cf_found = safe_get_value(df, "GO (Counterfactual Found)")
#     stop_cf_found = safe_get_value(df, "STOP (Counterfactual Found)")
#     go_no_cf = safe_get_value(df, "GO (No Counterfactual)")
#     stop_no_cf = safe_get_value(df, "STOP (No Counterfactual)")
#     time_taken = safe_get_value(df, "Total Time Taken (s)")
#     total_entries = safe_get_value(df, "Total Entries")

#     # Calculate percentages
#     go_cf_found_perc = (go_cf_found / total_go) * 100 if total_go > 0 else 0
#     stop_cf_found_perc = (stop_cf_found / total_stop) * 100 if total_stop > 0 else 0
#     go_no_cf_perc = (go_no_cf / total_go) * 100 if total_go > 0 else 0
#     stop_no_cf_perc = (stop_no_cf / total_stop) * 100 if total_stop > 0 else 0

#     combined_results.append({
#         "Method": method,
#         "GO (Counterfactual Found)": f"{go_cf_found} ({go_cf_found_perc:.2f}%)",
#         "STOP (Counterfactual Found)": f"{stop_cf_found} ({stop_cf_found_perc:.2f}%)",
#         "GO (No Counterfactual)": f"{go_no_cf} ({go_no_cf_perc:.2f}%)",
#         "STOP (No Counterfactual)": f"{stop_no_cf} ({stop_no_cf_perc:.2f}%)",
#         "Total Time Taken (s)": time_taken,
#         "Total Entries": total_entries,
#         "Total GO Cases": total_go,
#         "Total STOP Cases": total_stop,
#     })

# # Convert combined results into a DataFrame
# combined_table = pd.DataFrame(combined_results)

# # Save the combined table
# combined_table.to_csv("Projects/result_calculate/final_combined_results_summary.csv", index=False)

# # Display the combined table
# print("Final Combined Results with Number of Cases, Percentages, and Total GO/STOP Cases:")
# print(combined_table)

# Extract relevant metrics and combine results
combined_results = []

for method, df in zip(methods, [grid_df, object_detection_df, lime_latent_df, lime_image_df]):
    go_cf_found = safe_get_value(df, "GO (Counterfactual Found)")
    stop_cf_found = safe_get_value(df, "STOP (Counterfactual Found)")
    go_no_cf = safe_get_value(df, "GO (No Counterfactual)")
    stop_no_cf = safe_get_value(df, "STOP (No Counterfactual)")
    time_taken = safe_get_value(df, "Total Time Taken (s)")
    total_entries = safe_get_value(df, "Total Entries")
    total_percentage = safe_get_value(df, "Total Percentage (%)")

    # Calculate percentages relative to the method's total entries
    total_method_cases = go_cf_found + stop_cf_found + go_no_cf + stop_no_cf
    go_cf_found_perc = (go_cf_found / total_method_cases) * 100 if total_method_cases > 0 else 0
    stop_cf_found_perc = (stop_cf_found / total_method_cases) * 100 if total_method_cases > 0 else 0
    go_no_cf_perc = (go_no_cf / total_method_cases) * 100 if total_method_cases > 0 else 0
    stop_no_cf_perc = (stop_no_cf / total_method_cases) * 100 if total_method_cases > 0 else 0
    total_percentage = (total_method_cases / total_entries) * 100 if total_entries > 0 else 0

    combined_results.append({
        "Method": method,
        "GO (Counterfactual Found)": f"{go_cf_found} ({go_cf_found_perc:.2f}%)",
        "STOP (Counterfactual Found)": f"{stop_cf_found} ({stop_cf_found_perc:.2f}%)",
        "GO (No Counterfactual)": f"{go_no_cf} ({go_no_cf_perc:.2f}%)",
        "STOP (No Counterfactual)": f"{stop_no_cf} ({stop_no_cf_perc:.2f}%)",
        "Total Time Taken (s)": time_taken,
        "Total Entries": total_entries,
        "Total Percentage (%)": total_percentage,
    })

# Convert combined results into a DataFrame
combined_table = pd.DataFrame(combined_results)

# Save the combined table
combined_table.to_csv("Projects/result_calculate/final_combined_results_summary_relative.csv", index=False)

# Display the combined table
print("Final Combined Results with Percentages Summing to 100% Within Each Method:")
print(combined_table)
