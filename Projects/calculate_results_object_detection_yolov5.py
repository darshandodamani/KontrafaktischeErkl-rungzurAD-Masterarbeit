import pandas as pd

# File paths
original_train_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
original_test_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"
train_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv"
test_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"

# Load the original datasets to get total STOP and GO counts
train_original = pd.read_csv(original_train_csv)
test_original = pd.read_csv(original_test_csv)

# Load the object detection-based masking results
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Calculate total STOP and GO counts in the original datasets
total_train_stop = len(train_original[train_original['label'] == 'STOP'])
total_train_go = len(train_original[train_original['label'] == 'GO'])
total_test_stop = len(test_original[test_original['label'] == 'STOP'])
total_test_go = len(test_original[test_original['label'] == 'GO'])

def generate_summary_table(train_data, test_data):
    def summarize(data, total_stop, total_go):
        total_entries = len(data)
        total_time = data['Processing Time (s)'].sum()

        # Initialize counts
        summary = {
            "GO (Counterfactual Found)": len(data[(data['Prediction'] == 'GO') & (data['Counterfactual Found'] == True)]),
            "STOP (Counterfactual Found)": len(data[(data['Prediction'] == 'STOP') & (data['Counterfactual Found'] == True)]),
            "GO (No Counterfactual)": len(data[(data['Prediction'] == 'GO') & (data['Counterfactual Found'] == False)]),
            "STOP (No Counterfactual)": len(data[(data['Prediction'] == 'STOP') & (data['Counterfactual Found'] == False)]),
        }

        # Calculate percentages relative to total STOP and GO counts in the original dataset
        percentages = {
            "GO (Counterfactual Found)": (summary["GO (Counterfactual Found)"] / total_go) * 100,
            "STOP (Counterfactual Found)": (summary["STOP (Counterfactual Found)"] / total_stop) * 100,
            "GO (No Counterfactual)": (summary["GO (No Counterfactual)"] / total_go) * 100,
            "STOP (No Counterfactual)": (summary["STOP (No Counterfactual)"] / total_stop) * 100,
        }

        return summary, percentages, total_entries, total_time

    # Summarize train and test data
    train_summary, train_percentages, train_total_entries, train_total_time = summarize(train_data, total_train_stop, total_train_go)
    test_summary, test_percentages, test_total_entries, test_total_time = summarize(test_data, total_test_stop, total_test_go)

    # Create a combined summary table
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

    return summary_table

# Generate and display summary table
summary_table = generate_summary_table(train_data, test_data)

# Display the total STOP and GO counts from the original dataset
print("Original Dataset Counts:")
print(f"Total Train STOP: {total_train_stop}")
print(f"Total Train GO: {total_train_go}")
print(f"Total Test STOP: {total_test_stop}")
print(f"Total Test GO: {total_test_go}")

# Display the summary table
print(summary_table)

# Save the summary table to a CSV file
summary_table.to_csv("plots/object_detection_using_yolov5/object_detection_summary_results.csv", index=False)
