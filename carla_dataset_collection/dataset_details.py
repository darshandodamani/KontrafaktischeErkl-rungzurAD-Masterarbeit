import pandas as pd
import os

# File paths
train_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
test_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Check if files exist
if not os.path.exists(train_csv):
    raise FileNotFoundError(f"Train file not found: {train_csv}")
if not os.path.exists(test_csv):
    raise FileNotFoundError(f"Test file not found: {test_csv}")



# Read CSV files
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Replace 'Label' with the actual column name in your CSV
label_column = 'label'  # Update this to match your CSV structure

# Count STOP and GO for train dataset
train_counts = train_data[label_column].value_counts()

# Count STOP and GO for test dataset
test_counts = test_data[label_column].value_counts()

# Combine the results into a summary table
summary = pd.DataFrame({
    'Category': ['STOP', 'GO'],
    'Train Count': [train_counts.get('STOP', 0), train_counts.get('GO', 0)],
    'Test Count': [test_counts.get('STOP', 0), test_counts.get('GO', 0)],
})

# Add total counts
summary.loc[len(summary)] = ['Total', summary['Train Count'].sum(), summary['Test Count'].sum()]

# Display the summary
print(summary)

# Save the summary to a CSV file
summary.to_csv("plots/dataset_images/dataset_summary.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart visualization with count labels on top
def plot_summary_with_labels(summary):
    # Separate Train and Test counts
    categories = summary['Category'][:-1]  # Exclude 'Total' from the graph
    train_counts = summary['Train Count'][:-1]
    test_counts = summary['Test Count'][:-1]

    # Bar chart for Train and Test counts
    x = range(len(categories))
    width = 0.4

    plt.figure(figsize=(10, 6))
    bars_train = plt.bar(x, train_counts, width=width, label='Train Count', align='center')
    bars_test = plt.bar([p + width for p in x], test_counts, width=width, label='Test Count', align='center')

    # Add labels and legend
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('STOP and GO Distribution in Train and Test Datasets')
    plt.xticks([p + width / 2 for p in x], categories)
    plt.legend()

    # Add count numbers on top of each bar
    for bar in bars_train:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,  # Adjust position
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
    for bar in bars_test:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,  # Adjust position
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save and show the plot
    plt.savefig("plots/dataset_images/dataset_distribution_with_labels.png")

# Table visualization using Seaborn
def plot_table(summary):
    plt.figure(figsize=(6, 2))
    plt.axis('off')  # Turn off the axes
    sns.heatmap(
        summary.set_index('Category'),
        annot=True,
        fmt='g',
        cmap='Blues',
        cbar=False,
        linewidths=0.5
    )
    plt.title("Dataset Details Table")
    plt.tight_layout()

    # Save and show the table plot
    plt.savefig("plots/dataset_images/dataset_details_table.png")

# Plot the summary
plot_summary_with_labels(summary)
plot_table(summary)
