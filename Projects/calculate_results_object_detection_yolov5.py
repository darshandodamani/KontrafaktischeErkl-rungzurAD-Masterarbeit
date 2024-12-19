import pandas as pd

# Load CSV data
train_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv"
test_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"

# Read data
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

def generate_summary_table(train_data, test_data):
    def summarize(data):
        total_entries = len(data)
        total_time = data['Processing Time (s)'].sum()

        # Group by Prediction and Counterfactual Found
        summary = data.groupby(['Prediction', 'Counterfactual Found']).size().reset_index(name='Count')

        # Add missing combinations
        all_categories = [
            {'Prediction': 'GO', 'Counterfactual Found': True},
            {'Prediction': 'STOP', 'Counterfactual Found': True},
            {'Prediction': 'GO', 'Counterfactual Found': False},
            {'Prediction': 'STOP', 'Counterfactual Found': False},
        ]
        summary = pd.concat([summary, pd.DataFrame(all_categories)]).fillna(0)

        # Calculate percentages
        summary['Percentage (%)'] = (summary['Count'] / total_entries) * 100
        summary['Category'] = summary.apply(
            lambda row: f"{row['Prediction']} ({'Counterfactual Found' if row['Counterfactual Found'] else 'No Counterfactual'})",
            axis=1
        )
        summary = summary[['Category', 'Count', 'Percentage (%)']].groupby('Category').sum().reset_index()

        return summary, total_entries, total_time

    train_summary, train_total_entries, train_total_time = summarize(train_data)
    test_summary, test_total_entries, test_total_time = summarize(test_data)

    # Merge Train and Test Summaries
    merged_summary = train_summary.merge(
        test_summary, on='Category', how='outer', suffixes=(' Train', ' Test')
    ).fillna(0)

    # Add total time and entries
    merged_summary = pd.concat([
        merged_summary,
        pd.DataFrame({
            'Category': ['Total Time Taken (s)', 'Total Entries'],
            'Count Train': [train_total_time, train_total_entries],
            'Percentage (%) Train': ['-', '-'],
            'Count Test': [test_total_time, test_total_entries],
            'Percentage (%) Test': ['-', '-']
        })
    ])

    return merged_summary


# Generate and display summary table
summary_table = generate_summary_table(train_data, test_data)

# Save or display the table
summary_table.to_csv("evaluation_summary_table.csv", index=False)
print(summary_table)
