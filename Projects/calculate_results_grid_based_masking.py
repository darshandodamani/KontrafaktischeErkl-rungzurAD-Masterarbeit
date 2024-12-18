import pandas as pd

# Example CSV file paths (replace with your actual file paths)
train_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv"
test_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

# Load the train and test CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Define a function to calculate summary metrics
def summarize_results(df):
    total_time = df["Time Taken (s)"].sum()  # Total time taken
    total_entries = len(df)  # Total number of entries
    
    # Initialize counts
    summary = {
        "GO (Counterfactual Found)": 0,
        "STOP (Counterfactual Found)": 0,
        "GO (No Counterfactual)": 0,
        "STOP (No Counterfactual)": 0,
    }
    
    # Iterate through rows to classify results
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
    percentages = {key: (value / total_entries) * 100 for key, value in summary.items()}

    return summary, percentages, total_time, total_entries

# Summarize results for train and test datasets
train_summary, train_percentages, train_total_time, train_total_entries = summarize_results(train_df)
test_summary, test_percentages, test_total_time, test_total_entries = summarize_results(test_df)

# Create a summary table
summary_table = pd.DataFrame({
    "Category": [
        "GO (Counterfactual Found)",
        "STOP (Counterfactual Found)",
        "GO (No Counterfactual)",
        "STOP (No Counterfactual)",
        "Total Time Taken (s)",
        "Total Entries"
    ],
    "Train Count": [
        train_summary["GO (Counterfactual Found)"],
        train_summary["STOP (Counterfactual Found)"],
        train_summary["GO (No Counterfactual)"],
        train_summary["STOP (No Counterfactual)"],
        None,
        train_total_entries
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
        None,
        test_total_entries
    ],
    "Test Percentage (%)": [
        test_percentages["GO (Counterfactual Found)"],
        test_percentages["STOP (Counterfactual Found)"],
        test_percentages["GO (No Counterfactual)"],
        test_percentages["STOP (No Counterfactual)"],
        "-",
        "-"
    ],
})

# Add total time taken as a separate row
summary_table.loc[4, "Train Count"] = train_total_time
summary_table.loc[4, "Test Count"] = test_total_time

# Display the table
print(summary_table)

# Optionally save the table to a CSV file
summary_table.to_csv("grid_based_summary_results.csv", index=False)
