import pandas as pd

# Paths to the CSV files
train_csv = "plots/lime_plots/lime_based_counterfactual_results_train.csv"
test_csv = "plots/lime_plots/lime_based_counterfactual_results_test.csv"

# Load the CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Define a function to summarize results
def summarize_results(df):
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

    # Calculate percentages
    total = len(df)
    percentages = {key: (value / total) * 100 for key, value in summary.items()}

    return summary, percentages, total_time, total

# Summarize results for train and test datasets
train_summary, train_percentages, train_time, train_total = summarize_results(train_df)
test_summary, test_percentages, test_time, test_total = summarize_results(test_df)

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

# Display the table
print(final_table)

# Optionally save the table to a CSV file
final_table.to_csv("detailed_summary_results.csv", index=False)



