import pandas as pd

# Paths to CSV files
train_csv = "plots/lime_plots/lime_latent_masking_train_results.csv"
test_csv = "plots/lime_plots/lime_latent_masking_test_results.csv"

# Load the data
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Add a column to distinguish between train and test data
train_data["Dataset"] = "train"
test_data["Dataset"] = "test"

# Combine train and test data
data = pd.concat([train_data, test_data], ignore_index=True)

# Print column names to verify the correct column name
print("Column names:", data.columns)

# Find duplicates for debugging (optional)
duplicates = data[data.duplicated(subset=["Image File"], keep=False)]
print("Duplicate Entries:")
print(duplicates)

# Drop duplicates, keeping the first occurrence only
data_deduplicated = data.drop_duplicates(subset=["Image File"], keep="first")

# Split back into train and test based on the 'Dataset' column
train_deduplicated = data_deduplicated[data_deduplicated["Dataset"] == "train"].drop(columns=["Dataset"])
test_deduplicated = data_deduplicated[data_deduplicated["Dataset"] == "test"].drop(columns=["Dataset"])

# Save updated CSV files
train_deduplicated.to_csv("plots/lime_plots/lime_latent_masking_train_results_updated.csv", index=False)
test_deduplicated.to_csv("plots/lime_plots/lime_latent_masking_test_results_updated.csv", index=False)

# Print final stats
print("Updated train and test datasets saved without duplicates.")
print(f"Total rows in deduplicated train data: {len(train_deduplicated)}")
print(f"Total rows in deduplicated test data: {len(test_deduplicated)}")
