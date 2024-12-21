import pandas as pd

# Paths to the CSV files
train_csv = "plots/lime_plots/lime_based_counterfactual_results_train.csv"
test_csv = "plots/lime_plots/lime_based_counterfactual_results_test.csv"

# Load the CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Sum the 'Time Taken (s)' column for both datasets
total_train_time = train_df["Time Taken (s)"].sum()
total_test_time = test_df["Time Taken (s)"].sum()

# Calculate the combined total time
total_time = total_train_time + total_test_time

# Print the results
print(f"Total Time Taken for Train Dataset: {total_train_time:.2f} seconds")
print(f"Total Time Taken for Test Dataset: {total_test_time:.2f} seconds")
print(f"Combined Total Time: {total_time:.2f} seconds")

# in munutes
total_time_in_minutes_train = total_train_time / 60
total_time_in_minutes_test = total_test_time / 60
total_time_in_minutes = total_time / 60

# Print the results for minutes for train test and combined
print(f"Total Time Taken for Train Dataset: {total_time_in_minutes_train:.2f} seconds")
print(f"Total Time Taken for Test Dataset: {total_time_in_minutes_test:.2f} seconds")
print(f"Combined Total Time: {total_time_in_minutes:.2f} minutes")

