import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the responses CSV file
csv_file_path = "responses.csv"  # Update with correct path if needed
df = pd.read_csv(csv_file_path)

# Display basic statistics
summary_statistics = df.describe()
summary_statistics.to_csv("summary_statistics.csv")

# Define evaluation metrics
metrics = ["Interpretability", "Plausibility", "Visual Coherence"]

# Generate histograms for rating distributions
plt.figure(figsize=(12, 6))
df[metrics].hist(bins=5, figsize=(10, 6), edgecolor='black')
plt.suptitle("Distribution of Human Ratings for Counterfactual Explanations", fontsize=14)
plt.savefig("ratings_histogram.png")

# Correlation heatmap between evaluation metrics
plt.figure(figsize=(8,6))
sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Different Human Evaluation Metrics", fontsize=14)
plt.savefig("correlation_heatmap.png")

# Average ratings per chosen method (Bar Chart)
plt.figure(figsize=(10,6))
df.groupby("chosen_method")[metrics].mean().plot(kind="bar", figsize=(10,6), colormap="viridis")
plt.title("Average Human Ratings Per Chosen Method", fontsize=14)
plt.ylabel("Average Rating (1-5)")
plt.xlabel("Chosen Method")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.savefig("average_ratings_per_method.png")

# Boxplot for variability in responses
plt.figure(figsize=(12,6))
df_melted = df.melt(id_vars=["Chosen Method"], value_vars=metrics, var_name="Metric", value_name="Rating")
sns.boxplot(x="Metric", y="Rating", data=df_melted, palette="coolwarm")
plt.title("Variability in Human Ratings Across Metrics", fontsize=14)
plt.ylabel("Rating (1-5)")
plt.savefig("variability_in_ratings.png")

print("\nAnalysis Complete! Check the generated visualizations and summary statistics.")
