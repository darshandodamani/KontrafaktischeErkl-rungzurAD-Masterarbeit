import pandas as pd

# File paths for train and test results
csv_file_path_train = "plots/lime_on_images/lime_on_image_masking_train_results.csv"
csv_file_path_test = "plots/lime_on_images/lime_on_image_masking_test_results.csv"

# Load CSV files into dataframes
train_data = pd.read_csv(csv_file_path_train)
test_data = pd.read_csv(csv_file_path_test)

# Categories to summarize
categories = [
    "GO (Counterfactual Found)",
    "STOP (Counterfactual Found)",
    "GO (No Counterfactual)",
    "STOP (No Counterfactual)",
    "Total Time Taken (s)",
    "Total Entries"
]

# Function to calculate counts and percentages
def calculate_summary(data):
    # Total counts for each prediction
    total_go_predictions = data[data["Prediction"] == "GO"].shape[0]
    total_stop_predictions = data[data["Prediction"] == "STOP"].shape[0]

    # Counterfactual counts
    go_cf_found = data[(data["Prediction"] == "GO") & (data["Counterfactual Found"] == True)].shape[0]
    stop_cf_found = data[(data["Prediction"] == "STOP") & (data["Counterfactual Found"] == True)].shape[0]
    go_no_cf = data[(data["Prediction"] == "GO") & (data["Counterfactual Found"] == False)].shape[0]
    stop_no_cf = data[(data["Prediction"] == "STOP") & (data["Counterfactual Found"] == False)].shape[0]
    total_time = data["Time Taken (s)"].sum()
    total_entries = data.shape[0]
    
    # Calculate percentages based on total predictions
    go_cf_found_perc = (go_cf_found / total_go_predictions) * 100 if total_go_predictions else 0
    stop_cf_found_perc = (stop_cf_found / total_stop_predictions) * 100 if total_stop_predictions else 0
    go_no_cf_perc = (go_no_cf / total_go_predictions) * 100 if total_go_predictions else 0
    stop_no_cf_perc = (stop_no_cf / total_stop_predictions) * 100 if total_stop_predictions else 0

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
train_summary = calculate_summary(train_data)
test_summary = calculate_summary(test_data)

# Create a detailed summary table
summary_table = pd.DataFrame({
    "Category": categories,
    "Train Count": [train_summary[cat][0] for cat in categories],
    "Train Percentage (%)": [train_summary[cat][1] for cat in categories],
    "Test Count": [test_summary[cat][0] for cat in categories],
    "Test Percentage (%)": [test_summary[cat][1] for cat in categories]
})

# Display the detailed summary table
print("LIME on Image Masking Results Summary:")
print(summary_table.to_string(index=False))

# Optionally, save the summary table to a new CSV file
summary_table.to_csv("lime_on_image_masking_results_summary.csv", index=False)

# If you want to see specific details in the results:
# For example, how many total GO and STOP predictions were there?
total_go_train = train_data[train_data["Prediction"] == "GO"].shape[0]
total_stop_train = train_data[train_data["Prediction"] == "STOP"].shape[0]
print("\nDetailed Train Dataset Counts:")
print(f"Total GO Predictions: {total_go_train}")
print(f"Total STOP Predictions: {total_stop_train}")

total_go_test = test_data[test_data["Prediction"] == "GO"].shape[0]
total_stop_test = test_data[test_data["Prediction"] == "STOP"].shape[0]
print("\nDetailed Test Dataset Counts:")
print(f"Total GO Predictions: {total_go_test}")
print(f"Total STOP Predictions: {total_stop_test}")


