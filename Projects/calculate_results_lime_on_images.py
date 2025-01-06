#location: Projects/calculate_results_lime_on_images.py
import pandas as pd

# File paths
original_train_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
original_test_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"
csv_file_path_train = "plots/lime_on_images/lime_on_image_masking_train_results.csv"
csv_file_path_test = "plots/lime_on_images/lime_on_image_masking_test_results.csv"

# Load the original datasets to get total STOP and GO counts
train_original = pd.read_csv(original_train_csv)
test_original = pd.read_csv(original_test_csv)

# Load the LIME on Image Masking results
train_data = pd.read_csv(csv_file_path_train)
test_data = pd.read_csv(csv_file_path_test)

# Calculate total STOP and GO counts in the original datasets
total_train_stop = len(train_original[train_original['label'] == 'STOP'])
total_train_go = len(train_original[train_original['label'] == 'GO'])
total_test_stop = len(test_original[test_original['label'] == 'STOP'])
total_test_go = len(test_original[test_original['label'] == 'GO'])

# Function to calculate counts and percentages
def calculate_summary(data, total_stop, total_go):
    # Counterfactual counts
    go_cf_found = data[(data["Prediction"] == "GO") & (data["Counterfactual Found"] == True)].shape[0]
    stop_cf_found = data[(data["Prediction"] == "STOP") & (data["Counterfactual Found"] == True)].shape[0]
    go_no_cf = data[(data["Prediction"] == "GO") & (data["Counterfactual Found"] == False)].shape[0]
    stop_no_cf = data[(data["Prediction"] == "STOP") & (data["Counterfactual Found"] == False)].shape[0]
    total_time = data["Time Taken (s)"].sum()
    total_entries = data.shape[0]
    
    # Calculate percentages relative to original STOP and GO counts
    go_cf_found_perc = (go_cf_found / total_go) * 100 if total_go else 0
    stop_cf_found_perc = (stop_cf_found / total_stop) * 100 if total_stop else 0
    go_no_cf_perc = (go_no_cf / total_go) * 100 if total_go else 0
    stop_no_cf_perc = (stop_no_cf / total_stop) * 100 if total_stop else 0

    # Return dictionary with counts and percentages
    return {
        "GO (Counterfactual Found)": [go_cf_found, go_cf_found_perc],
        "STOP (Counterfactual Found)": [stop_cf_found, stop_cf_found_perc],
        "GO (No Counterfactual)": [go_no_cf, go_no_cf_perc],
        "STOP (No Counterfactual)": [stop_no_cf, stop_no_cf_perc],
        "Total Time Taken (s)": [total_time, "-"],
        "Total Entries": [total_entries, "-"]
    }

# Calculate summaries for train and test data
train_summary = calculate_summary(train_data, total_train_stop, total_train_go)
test_summary = calculate_summary(test_data, total_test_stop, total_test_go)

# Categories to summarize
categories = [
    "GO (Counterfactual Found)",
    "STOP (Counterfactual Found)",
    "GO (No Counterfactual)",
    "STOP (No Counterfactual)",
    "Total Time Taken (s)",
    "Total Entries"
]

# Create a detailed summary table
summary_table = pd.DataFrame({
    "Category": categories,
    "Train Count": [train_summary[cat][0] for cat in categories],
    "Train Percentage (%)": [train_summary[cat][1] for cat in categories],
    "Test Count": [test_summary[cat][0] for cat in categories],
    "Test Percentage (%)": [test_summary[cat][1] for cat in categories]
})

# Display the total STOP and GO counts from the original dataset
print("Original Dataset Counts:")
print(f"Total Train STOP: {total_train_stop}")
print(f"Total Train GO: {total_train_go}")
print(f"Total Test STOP: {total_test_stop}")
print(f"Total Test GO: {total_test_go}")

# Display the detailed summary table
print("\nLIME on Image Masking Results Summary:")
print(summary_table.to_string(index=False))

# Optionally save the table to a CSV file
summary_table.to_csv("plots/lime_on_images/lime_on_image_masking_results_summary.csv", index=False)
