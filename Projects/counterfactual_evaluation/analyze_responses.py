import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the responses
df = pd.read_csv("responses.csv")

# Display basic statistics
print("\n Summary Statistics:\n")
print(df.describe())

# Show first few rows
print("\n First Few Rows of Data:\n")
print(df.head())

# Define columns to analyze
metrics = ["Interpretability", "Plausibility", "Actionability", "Trust in AI", "Visual Coherence"]

# Distribution of Human Ratings (Histograms)
plt.figure(figsize=(12, 6))
df[metrics].hist(bins=5, figsize=(10, 6), edgecolor='black')
plt.suptitle("Distribution of Human Ratings for Counterfactual Explanations", fontsize=14)
plt.savefig("ratings_histogram.png")

# Correlation Between Different Evaluation Metrics
plt.figure(figsize=(8,6))
sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Different Human Evaluation Metrics", fontsize=14)
plt.savefig("correlation_heatmap.png")

# Average Ratings Per Image Pair (Bar Chart)
plt.figure(figsize=(10,6))
df.groupby("Image Pair")[metrics].mean().plot(kind="bar", figsize=(10,6), colormap="viridis")
plt.title("Average Human Ratings Per Image Pair", fontsize=14)
plt.ylabel("Average Rating (1-5)")
plt.xlabel("Image Pair")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.savefig("average_ratings_per_image_pair.png")

# Boxplot for Variability in Responses
plt.figure(figsize=(12,6))
df_melted = df.melt(id_vars=["Image Pair"], value_vars=metrics, var_name="Metric", value_name="Rating")
sns.boxplot(x="Metric", y="Rating", data=df_melted, palette="coolwarm")
plt.title("Variability in Human Ratings Across Metrics", fontsize=14)
plt.ylabel("Rating (1-5)")
plt.savefig("variability_in_ratings.png")

# add summary statistics into the report csv file
summary_statistics = df.describe().T
summary_statistics.to_csv("summary_statistics.csv")



# Print a message to indicate the analysis is complete
print("\n Analysis Complete! Check the visualizations.")
