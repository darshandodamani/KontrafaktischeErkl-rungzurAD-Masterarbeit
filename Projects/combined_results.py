import pandas as pd

# File paths to individual summaries
grid_csv = "plots/grid_based_masking_images/grid_based_summary_results.csv"
object_detection_csv = "plots/object_detection_using_yolov5/object_detection_summary_results.csv"
lime_latent_csv = "plots/lime_plots/detailed_summary_results.csv"
lime_image_csv = "plots/lime_on_images/lime_on_image_masking_results_summary.csv"

# Load all CSV files
grid_df = pd.read_csv(grid_csv)
object_detection_df = pd.read_csv(object_detection_csv)
lime_latent_df = pd.read_csv(lime_latent_csv)
lime_image_df = pd.read_csv(lime_image_csv)

# Total GO and STOP counts from the original dataset
total_go = 4959 + 1262  # Train GO + Test GO
total_stop = 4728 + 1160  # Train STOP + Test STOP

# Correct calculation of percentages based on total GO/STOP cases
def combine_train_test(df, total_go, total_stop):
    combined_count = df["Train Count"].fillna(0) + df["Test Count"].fillna(0)
    combined_percentage = combined_count.copy()
    # Calculate percentages for GO and STOP based on original dataset counts
    combined_percentage.iloc[0] = (combined_count.iloc[0] / total_go) * 100  # GO (CF Found)
    combined_percentage.iloc[1] = (combined_count.iloc[1] / total_stop) * 100  # STOP (CF Found)
    combined_percentage.iloc[2] = (combined_count.iloc[2] / total_go) * 100  # GO (No CF)
    combined_percentage.iloc[3] = (combined_count.iloc[3] / total_stop) * 100  # STOP (No CF)
    return combined_count, combined_percentage


# Process all datasets
methods = ["LIME-Based", "Grid-Based", "Object Detection-Based", "LIME on Image Masking"]
datasets = [lime_latent_df, grid_df, object_detection_df, lime_image_df]

combined_results = []
for method, df in zip(methods, datasets):
    # Calculate combined metrics
    combined_count, combined_percentage = combine_train_test(df, total_go, total_stop)
    time_taken = df[df["Category"] == "Total Time Taken (s)"]["Train Count"].values[0] + df[df["Category"] == "Total Time Taken (s)"]["Test Count"].values[0]
    total_entries = df[df["Category"] == "Total Entries"]["Train Count"].values[0] + df[df["Category"] == "Total Entries"]["Test Count"].values[0]
    combined_results.append({
        "Method": method,
        "GO (Counterfactual Found)": f"{combined_percentage[0]:.2f}% ({combined_count[0]} cases)",
        "STOP (Counterfactual Found)": f"{combined_percentage[1]:.2f}% ({combined_count[1]} cases)",
        "GO (No Counterfactual)": f"{combined_percentage[2]:.2f}% ({combined_count[2]} cases)",
        "STOP (No Counterfactual)": f"{combined_percentage[3]:.2f}% ({combined_count[3]} cases)",
        "Total Time Taken (s)": time_taken,
        "Total Entries": total_entries,
        "Total GO Cases": total_go,
        "Total STOP Cases": total_stop,
    })

# Convert to DataFrame
summary_df = pd.DataFrame(combined_results)

# Save the combined results
summary_df.to_csv("final_combined_results_with_total_go_stop.csv", index=False)

# Display the final results
print("Final Combined Results with Total GO and STOP Cases:")
print(summary_df.to_string(index=False))
