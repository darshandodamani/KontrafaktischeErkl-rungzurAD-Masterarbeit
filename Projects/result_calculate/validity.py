import pandas as pd

# Paths to result CSV files
grid_csv = "plots/grid_based_masking_results.csv"
object_detection_csv = "plots/object_detection_masking_results.csv"
lime_on_images_csv = "plots/lime_on_image_masking_results.csv"
lime_on_latent_csv = "plots/lime_on_latent_masking_results.csv"

# Load datasets
grid_df = pd.read_csv(grid_csv)
object_detection_df = pd.read_csv(object_detection_csv)
lime_image_df = pd.read_csv(lime_on_images_csv)
lime_latent_df = pd.read_csv(lime_on_latent_csv)

# Function to calculate validity
def calculate_validity(df, method_name):
    total_counterfactuals = len(df)
    successful_counterfactuals = df[df["Counterfactual Found"] == True].shape[0]
    validity = (successful_counterfactuals / total_counterfactuals) * 100 if total_counterfactuals > 0 else 0
    
    return {"Method": method_name, "Total Counterfactuals": total_counterfactuals, 
            "Successful Counterfactuals": successful_counterfactuals, "Validity (%)": round(validity, 2)}

# Compute validity for each method
validity_results = [
    calculate_validity(grid_df, "Grid-Based Masking"),
    calculate_validity(object_detection_df, "Object Detection Masking"),
    calculate_validity(lime_image_df, "LIME on Images"),
    calculate_validity(lime_latent_df, "LIME on Latent Features")
]

# Convert to DataFrame and save as CSV
validity_df = pd.DataFrame(validity_results)
validity_csv_path = "plots/validity_summary.csv"
validity_df.to_csv(validity_csv_path, index=False)

# Print validity results
print("\nValidity Summary:")
print(validity_df)
print(f"\nValidity summary saved to: {validity_csv_path}")
