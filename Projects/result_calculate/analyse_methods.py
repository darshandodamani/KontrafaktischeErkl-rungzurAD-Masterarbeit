import os
import pandas as pd
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

# Ensure output directories exist
os.makedirs("plots/venn_diagram", exist_ok=True)
os.makedirs("Projects/result_calculate", exist_ok=True)

# Load data
grid_df = pd.read_csv("plots/grid_based_masking_results.csv")
object_detection_df = pd.read_csv("plots/object_detection_masking_results.csv")
lime_latent_df = pd.read_csv("plots/lime_on_latent_masking_results.csv")
lime_image_df = pd.read_csv("plots/lime_on_image_masking_results.csv")

# Filter for Counterfactual Found == True
grid_ce = set(grid_df[grid_df["Counterfactual Found"] == True]["Image File"])
object_detection_ce = set(object_detection_df[object_detection_df["Counterfactual Found"] == True]["Image File"])
lime_latent_ce = set(lime_latent_df[lime_latent_df["Counterfactual Found"] == True]["Image File"])
lime_image_ce = set(lime_image_df[lime_image_df["Counterfactual Found"] == True]["Image File"])

# Combine all unique images across methods
all_images = (
    set(grid_df["Image File"])
    | set(object_detection_df["Image File"])
    | set(lime_latent_df["Image File"])
    | set(lime_image_df["Image File"])
)

# Generate the table
table = []
for image in all_images:
    table.append({
        "Image File": image,
        "Grid": "True" if image in grid_ce else "False",
        "LIME on Latent Features": "True" if image in lime_latent_ce else "False",
        "LIME on Images": "True" if image in lime_image_ce else "False",
        "Object Detection": "True" if image in object_detection_ce else "False",
    })

table_df = pd.DataFrame(table)
table_csv_path = "Projects/result_calculate/method_comparison_table.csv"
table_df.to_csv(table_csv_path, index=False)
print(f"Table saved as '{table_csv_path}'.")

# Use venny4py for the Venn diagram
sets = {
    "Grid-Based Masking": grid_ce,
    "LIME on Latent Features": lime_latent_ce,
    "LIME on Images": lime_image_ce,
    "Object Detection-Based": object_detection_ce,
}

venny4py(sets=sets)

# Save the diagram as a PNG
venn_diagram_path = "plots/venn_diagram/venn_diagram_four_methods.png"
plt.title("Venn Diagram of Counterfactual Explanation Coverage")
plt.savefig(venn_diagram_path)
print(f"Venn diagram saved as '{venn_diagram_path}'.")

# Summary statistics for individual methods
print("\nSummary of Counterfactual Explanation Coverage:")
print(f"Total Images: {len(all_images)}")
print(f"Images Explained by Grid-Based Masking: {len(grid_ce)}")
print(f"Images Explained by LIME on Latent Features: {len(lime_latent_ce)}")
print(f"Images Explained by LIME on Images: {len(lime_image_ce)}")
print(f"Images Explained by Object Detection-Based: {len(object_detection_ce)}")

# Overlaps for exactly two methods
explained_by_two_methods = {
    "Grid-Based & LIME on Latent Features": len((grid_ce & lime_latent_ce) - lime_image_ce - object_detection_ce),
    "Grid-Based & LIME on Images": len((grid_ce & lime_image_ce) - lime_latent_ce - object_detection_ce),
    "Grid-Based & Object Detection-Based": len((grid_ce & object_detection_ce) - lime_latent_ce - lime_image_ce),
    "LIME on Latent Features & LIME on Images": len((lime_latent_ce & lime_image_ce) - grid_ce - object_detection_ce),
    "LIME on Latent Features & Object Detection-Based": len((lime_latent_ce & object_detection_ce) - grid_ce - lime_image_ce),
    "LIME on Images & Object Detection-Based": len((lime_image_ce & object_detection_ce) - grid_ce - lime_latent_ce),
}

print("\nImages Explained by Exactly Two Methods:")
for combination, count in explained_by_two_methods.items():
    print(f"{combination}: {count} images")

# Overlaps for exactly three methods
explained_by_three_methods = {
    "Grid-Based, LIME on Latent Features & LIME on Images": len((grid_ce & lime_latent_ce & lime_image_ce) - object_detection_ce),
    "Grid-Based, LIME on Latent Features & Object Detection-Based": len((grid_ce & lime_latent_ce & object_detection_ce) - lime_image_ce),
    "Grid-Based, LIME on Images & Object Detection-Based": len((grid_ce & lime_image_ce & object_detection_ce) - lime_latent_ce),
    "LIME on Latent Features, LIME on Images & Object Detection-Based": len((lime_latent_ce & lime_image_ce & object_detection_ce) - grid_ce),
}

print("\nImages Explained by Exactly Three Methods:")
for combination, count in explained_by_three_methods.items():
    print(f"{combination}: {count} images")

# Images explained by all four methods
explained_by_all_methods = len(grid_ce & lime_latent_ce & lime_image_ce & object_detection_ce)
print(f"\nImages Explained by All Four Methods: {explained_by_all_methods} images")

# Images not explained by any method
not_explained = all_images - (grid_ce | lime_latent_ce | lime_image_ce | object_detection_ce)
print(f"\nImages Not Explained by Any Method: {len(not_explained)} images")
