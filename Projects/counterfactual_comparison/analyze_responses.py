import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
file_path = "Projects/counterfactual_comparison/responses.csv"  # Ensure the path to your responses file is correct
df = pd.read_csv(file_path)

# Inspect dataset
print("Dataset Overview:\n", df.head())  # Check structure
print("Detected columns:", df.columns.tolist())

# **Step 1: Rename Columns for Clarity**
method_mapping = {
    "Counterfactual_1_Interpretability": "Grid-Based Masking_Interpretability",
    "Counterfactual_1_Plausibility": "Grid-Based Masking_Plausibility",
    "Counterfactual_1_VisualCoherence": "Grid-Based Masking_VisualCoherence",
    
    "Counterfactual_2_Interpretability": "LIME on Images_Interpretability",
    "Counterfactual_2_Plausibility": "LIME on Images_Plausibility",
    "Counterfactual_2_VisualCoherence": "LIME on Images_VisualCoherence",
    
    "Counterfactual_3_Interpretability": "LIME on Latent Features_Interpretability",
    "Counterfactual_3_Plausibility": "LIME on Latent Features_Plausibility",
    "Counterfactual_3_VisualCoherence": "LIME on Latent Features_VisualCoherence",
}

# Rename the columns
df.rename(columns=method_mapping, inplace=True)

# **Step 2: Convert Wide Format to Long Format**
df_long = df.melt(id_vars=["Image"], var_name="Method_Criterion", value_name="Rating")

# **Step 3: Split Method and Criterion**
df_long["Method"] = df_long["Method_Criterion"].apply(lambda x: x.split("_")[0])  
df_long["Criterion"] = df_long["Method_Criterion"].apply(lambda x: x.split("_")[1])  

# Drop the original combined column
df_long.drop(columns=["Method_Criterion"], inplace=True)

# **Step 4: Convert ratings to numeric & remove NaNs**
df_long["Rating"] = pd.to_numeric(df_long["Rating"], errors="coerce")
df_long.dropna(inplace=True)

# **Step 5: Aggregate duplicate values**
df_summary = df_long.groupby(["Method", "Criterion"])["Rating"].mean().reset_index()

# **Step 6: Pivot for Visualization**
df_pivot = df_summary.pivot(index="Criterion", columns="Method", values="Rating")

# Create output directory
output_dir = "plots/counterfactual_comparison/"
os.makedirs(output_dir, exist_ok=True)

### ** Bar Plot**
plt.figure(figsize=(10, 6))
df_pivot.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")

plt.title("User Evaluation of Counterfactual Explanation Methods")
plt.xlabel("Evaluation Criteria")
plt.ylabel("Average Rating (1-5)")
plt.xticks(rotation=45)
plt.legend(title="Method")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save bar plot
bar_plot_path = os.path.join(output_dir, "bar_plot_user_evaluations.png")
plt.savefig(bar_plot_path)
plt.show()

print(f" Bar plot saved: {bar_plot_path}")

### ** Heatmap**
plt.figure(figsize=(8, 5))
sns.heatmap(df_pivot, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")

plt.title("User Evaluation Heatmap")
plt.xlabel("Method")
plt.ylabel("Evaluation Criterion")

# Save heatmap
heatmap_path = os.path.join(output_dir, "heatmap_user_evaluations.png")
plt.savefig(heatmap_path)
plt.show()

print(f" Heatmap saved: {heatmap_path}")