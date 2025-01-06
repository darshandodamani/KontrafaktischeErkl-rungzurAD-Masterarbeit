# location: Projects/calculate_results_latent_features.py
import pandas as pd

# File paths
original_train_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
original_test_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"
train_csv = "plots/lime_plots/lime_based_counterfactual_results_train.csv"
test_csv = "plots/lime_plots/lime_based_counterfactual_results_test.csv"

# Load the original datasets to get total STOP and GO counts
train_original = pd.read_csv(original_train_csv)
test_original = pd.read_csv(original_test_csv)

# Load the LIME-based results
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Calculate total STOP and GO counts in the original datasets
total_train_stop = len(train_original[train_original['label'] == 'STOP'])
total_train_go = len(train_original[train_original['label'] == 'GO'])
total_test_stop = len(test_original[test_original['label'] == 'STOP'])
total_test_go = len(test_original[test_original['label'] == 'GO'])

# Define a function to summarize results
def summarize_results(df, total_stop, total_go):
    # Total time taken
    total_time = df["Time Taken (s)"].sum()

    # Initialize counters
    summary = {
        "GO (Counterfactual Found)": 0,
        "STOP (Counterfactual Found)": 0,
        "GO (No Counterfactual)": 0,
        "STOP (No Counterfactual)": 0,
    }

    # Iterate through rows and count occurrences
    for _, row in df.iterrows():
        prediction = row["Prediction"]
        counterfactual_found = row["Counterfactual Found"]

        if prediction == "GO" and counterfactual_found:
            summary["GO (Counterfactual Found)"] += 1
        elif prediction == "STOP" and counterfactual_found:
            summary["STOP (Counterfactual Found)"] += 1
        elif prediction == "GO" and not counterfactual_found:
            summary["GO (No Counterfactual)"] += 1
        elif prediction == "STOP" and not counterfactual_found:
            summary["STOP (No Counterfactual)"] += 1

    # Calculate percentages relative to total STOP and GO counts
    percentages = {
        "GO (Counterfactual Found)": (summary["GO (Counterfactual Found)"] / total_go) * 100,
        "STOP (Counterfactual Found)": (summary["STOP (Counterfactual Found)"] / total_stop) * 100,
        "GO (No Counterfactual)": (summary["GO (No Counterfactual)"] / total_go) * 100,
        "STOP (No Counterfactual)": (summary["STOP (No Counterfactual)"] / total_stop) * 100,
    }

    return summary, percentages, total_time, len(df)

# Summarize results for train and test datasets
train_summary, train_percentages, train_time, train_total = summarize_results(train_df, total_train_stop, total_train_go)
test_summary, test_percentages, test_time, test_total = summarize_results(test_df, total_test_stop, total_test_go)

# Create a detailed summary table
summary_table = pd.DataFrame({
    "Category": [
        "GO (Counterfactual Found)",
        "STOP (Counterfactual Found)",
        "GO (No Counterfactual)",
        "STOP (No Counterfactual)"
    ],
    "Train Count": [train_summary[key] for key in train_summary.keys()],
    "Train Percentage (%)": [train_percentages[key] for key in train_percentages.keys()],
    "Test Count": [test_summary[key] for key in test_summary.keys()],
    "Test Percentage (%)": [test_percentages[key] for key in test_percentages.keys()],
})

# Add total time and dataset sizes as additional rows
total_summary = pd.DataFrame({
    "Category": ["Total Time Taken (s)", "Total Entries"],
    "Train Count": [train_time, train_total],
    "Train Percentage (%)": ["-", "-"],
    "Test Count": [test_time, test_total],
    "Test Percentage (%)": ["-", "-"]
})

# Combine the detailed table and total summary
final_table = pd.concat([summary_table, total_summary], ignore_index=True)

# Display the total STOP and GO counts from the original dataset
print("Original Dataset Counts:")
print(f"Total Train STOP: {total_train_stop}")
print(f"Total Train GO: {total_train_go}")
print(f"Total Test STOP: {total_test_stop}")
print(f"Total Test GO: {total_test_go}")

# Display the table
print(final_table)

# Optionally save the table to a CSV file
final_table.to_csv("plots/lime_plots/detailed_summary_results.csv", index=False)
