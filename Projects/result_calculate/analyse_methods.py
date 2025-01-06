import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib_venn import venn3  # Support for 3 sets only

# File paths to train and test CSV files for each method
grid_train_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv"
grid_test_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

object_detection_train_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv"
object_detection_test_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"

lime_latent_train_csv = "plots/lime_plots/lime_based_counterfactual_results_train.csv"
lime_latent_test_csv = "plots/lime_plots/lime_based_counterfactual_results_test.csv"

lime_image_train_csv = "plots/lime_on_images/lime_on_image_masking_train_results.csv"
lime_image_test_csv = "plots/lime_on_images/lime_on_image_masking_test_results.csv"

# Load and combine train and test data for each method
grid_df = pd.concat([pd.read_csv(grid_train_csv), pd.read_csv(grid_test_csv)], ignore_index=True)
object_detection_df = pd.concat([pd.read_csv(object_detection_train_csv), pd.read_csv(object_detection_test_csv)], ignore_index=True)
lime_latent_df = pd.concat([pd.read_csv(lime_latent_train_csv), pd.read_csv(lime_latent_test_csv)], ignore_index=True)
lime_image_df = pd.concat([pd.read_csv(lime_image_train_csv), pd.read_csv(lime_image_test_csv)], ignore_index=True)

# Filter for Counterfactual Found == True
grid_ce = set(grid_df[grid_df["Counterfactual Found"] == True]["Image File"])
object_detection_ce = set(object_detection_df[object_detection_df["Counterfactual Found"] == True]["Image File"])
lime_latent_ce = set(lime_latent_df[lime_latent_df["Counterfactual Found"] == True]["Image File"])
lime_image_ce = set(lime_image_df[lime_image_df["Counterfactual Found"] == True]["Image File"])

# Summarize overlaps
explained_by_all = grid_ce & object_detection_ce & lime_latent_ce & lime_image_ce
explained_by_three = (
    (grid_ce & object_detection_ce & lime_latent_ce)
    | (grid_ce & object_detection_ce & lime_image_ce)
    | (grid_ce & lime_latent_ce & lime_image_ce)
    | (object_detection_ce & lime_latent_ce & lime_image_ce)
) - explained_by_all
explained_by_two = (
    (grid_ce & object_detection_ce)
    | (grid_ce & lime_latent_ce)
    | (grid_ce & lime_image_ce)
    | (object_detection_ce & lime_latent_ce)
    | (object_detection_ce & lime_image_ce)
    | (lime_latent_ce & lime_image_ce)
) - explained_by_all - explained_by_three
explained_by_one = (
    grid_ce | object_detection_ce | lime_latent_ce | lime_image_ce
) - explained_by_all - explained_by_three - explained_by_two
not_explained = set(grid_df["Image File"]) | set(object_detection_df["Image File"]) | set(lime_latent_df["Image File"]) | set(lime_image_df["Image File"])
not_explained -= grid_ce | object_detection_ce | lime_latent_ce | lime_image_ce

# Print textual representation
summary = {
    "Explained by All Methods": len(explained_by_all),
    "Explained by Three Methods": len(explained_by_three),
    "Explained by Two Methods": len(explained_by_two),
    "Explained by One Method": len(explained_by_one),
    "Not Explained by Any Method": len(not_explained),
}
print("Summary of Counterfactual Explanation:")
for key, value in summary.items():
    print(f"{key}: {value}")

# Visualization using venn3 for three main overlaps
plt.figure(figsize=(8, 8))
venn = venn3(
    subsets=(
        len(grid_ce - object_detection_ce - lime_latent_ce),
        len(object_detection_ce - grid_ce - lime_latent_ce),
        len(grid_ce & object_detection_ce - lime_latent_ce),
        len(lime_latent_ce - grid_ce - object_detection_ce),
        len(grid_ce & lime_latent_ce - object_detection_ce),
        len(object_detection_ce & lime_latent_ce - grid_ce),
        len(grid_ce & object_detection_ce & lime_latent_ce),
    ),
    set_labels=("Grid-Based", "Object Detection", "LIME-Based Latent"),
)
plt.title("Counterfactual Explanation Overlap (3 Methods)")
plt.savefig("Projects/result_calculate/venn_diagram_three_methods.png")
