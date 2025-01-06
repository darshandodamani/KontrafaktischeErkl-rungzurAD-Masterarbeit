import pandas as pd

# File paths
train_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv"
test_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

# Original dataset totals
train_dataset_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
test_dataset_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Load original dataset to calculate total GO and STOP cases
train_data = pd.read_csv(train_dataset_csv)
test_data = pd.read_csv(test_dataset_csv)
total_go = train_data[train_data["label"] == "GO"].shape[0] + test_data[test_data["label"] == "GO"].shape[0]
total_stop = train_data[train_data["label"] == "STOP"].shape[0] + test_data[test_data["label"] == "STOP"].shape[0]

# Load train and test CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Function to summarize results
def summarize_results(df, total_go, total_stop):
    # Initialize counts
    summary = {
        "GO (Counterfactual Found)": 0,
        "STOP (Counterfactual Found)": 0,
        "GO (No Counterfactual)": 0,
        "STOP (No Counterfactual)": 0,
        "Total Time Taken (s)": 0,
        "Total Entries": len(df),
    }

    # Iterate through rows and categorize
    for _, row in df.iterrows():
        prediction = row["Prediction"]
        counterfactual_found = row["Counterfactual Found"]
        time_taken = row["Time Taken (s)"]

        # Increment time
        summary["Total Time Taken (s)"] += time_taken

        # Classify the row
        if prediction == "GO" and counterfactual_found:
            summary["GO (Counterfactual Found)"] += 1
        elif prediction == "STOP" and counterfactual_found:
            summary["STOP (Counterfactual Found)"] += 1
        elif prediction == "GO" and not counterfactual_found:
            summary["GO (No Counterfactual)"] += 1
        elif prediction == "STOP" and not counterfactual_found:
            summary["STOP (No Counterfactual)"] += 1

    # Calculate percentages
    percentages = {
        "GO (Counterfactual Found)": (summary["GO (Counterfactual Found)"] / total_go) * 100,
        "STOP (Counterfactual Found)": (summary["STOP (Counterfactual Found)"] / total_stop) * 100,
        "GO (No Counterfactual)": (summary["GO (No Counterfactual)"] / total_go) * 100,
        "STOP (No Counterfactual)": (summary["STOP (No Counterfactual)"] / total_stop) * 100,
    }

    return summary, percentages

# Summarize results for train and test datasets
train_summary, train_percentages = summarize_results(train_df, total_go, total_stop)
test_summary, test_percentages = summarize_results(test_df, total_go, total_stop)

# Combine train and test summaries
combined_summary = {key: train_summary[key] + test_summary[key] for key in train_summary.keys()}
combined_percentages = {key: (combined_summary[key] / total_go if "GO" in key else combined_summary[key] / total_stop) * 100
                        for key in combined_summary.keys() if key not in ["Total Time Taken (s)", "Total Entries"]}

# Add total time and entries for combined
combined_summary["Total Time Taken (s)"] = train_summary["Total Time Taken (s)"] + test_summary["Total Time Taken (s)"]
combined_summary["Total Entries"] = train_summary["Total Entries"] + test_summary["Total Entries"]

# Create a DataFrame for the results
summary_table = pd.DataFrame({
    "Metric": ["GO (Counterfactual Found)", "STOP (Counterfactual Found)", "GO (No Counterfactual)", "STOP (No Counterfactual)", "Total Time Taken (s)", "Total Entries"],
    "Train Count": [
        train_summary["GO (Counterfactual Found)"],
        train_summary["STOP (Counterfactual Found)"],
        train_summary["GO (No Counterfactual)"],
        train_summary["STOP (No Counterfactual)"],
        train_summary["Total Time Taken (s)"],
        train_summary["Total Entries"]
    ],
    "Train Percentage (%)": [
        train_percentages["GO (Counterfactual Found)"],
        train_percentages["STOP (Counterfactual Found)"],
        train_percentages["GO (No Counterfactual)"],
        train_percentages["STOP (No Counterfactual)"],
        "-",
        "-"
    ],
    "Test Count": [
        test_summary["GO (Counterfactual Found)"],
        test_summary["STOP (Counterfactual Found)"],
        test_summary["GO (No Counterfactual)"],
        test_summary["STOP (No Counterfactual)"],
        test_summary["Total Time Taken (s)"],
        test_summary["Total Entries"]
    ],
    "Test Percentage (%)": [
        test_percentages["GO (Counterfactual Found)"],
        test_percentages["STOP (Counterfactual Found)"],
        test_percentages["GO (No Counterfactual)"],
        test_percentages["STOP (No Counterfactual)"],
        "-",
        "-"
    ],
    "Combined Count": [
        combined_summary["GO (Counterfactual Found)"],
        combined_summary["STOP (Counterfactual Found)"],
        combined_summary["GO (No Counterfactual)"],
        combined_summary["STOP (No Counterfactual)"],
        combined_summary["Total Time Taken (s)"],
        combined_summary["Total Entries"]
    ],
    "Combined Percentage (%)": [
        combined_percentages["GO (Counterfactual Found)"],
        combined_percentages["STOP (Counterfactual Found)"],
        combined_percentages["GO (No Counterfactual)"],
        combined_percentages["STOP (No Counterfactual)"],
        "-",
        "-"
    ]
})

# Display the table
print(summary_table)

# Save the table to a CSV file
summary_table.to_csv("Projects/result_calculate/grid_based_results_summary.csv", index=False)
