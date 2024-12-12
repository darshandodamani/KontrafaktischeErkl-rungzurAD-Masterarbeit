import pandas as pd
import matplotlib.pyplot as plt

def evaluate_results_with_visualization(csv_file, dataset_name, method_name, plot_file):
    """
    Evaluate the masking results, calculate percentages, and visualize results.
    """
    df = pd.read_csv(csv_file)
    total_images = len(df)

    # Count True and False for GO and STOP after masking
    go_true = len(df[(df["Prediction"] == "GO") & (df["Counterfactual Found"] == True)])
    go_false = len(df[(df["Prediction"] == "GO") & (df["Counterfactual Found"] == False)])
    stop_true = len(df[(df["Prediction"] == "STOP") & (df["Counterfactual Found"] == True)])
    stop_false = len(df[(df["Prediction"] == "STOP") & (df["Counterfactual Found"] == False)])

    # Calculate percentages
    go_true_percentage = (go_true / total_images) * 100
    go_false_percentage = (go_false / total_images) * 100
    stop_true_percentage = (stop_true / total_images) * 100
    stop_false_percentage = (stop_false / total_images) * 100

    print(f"{method_name} Results for {dataset_name} Dataset:")
    print(f"Total Images: {total_images}")
    print(f"GO - True: {go_true} ({go_true_percentage:.2f}%), False: {go_false} ({go_false_percentage:.2f}%)")
    print(f"STOP - True: {stop_true} ({stop_true_percentage:.2f}%), False: {stop_false} ({stop_false_percentage:.2f}%)")
    print("-" * 50)

    # Visualization
    categories = ["GO - True", "GO - False", "STOP - True", "STOP - False"]
    values = [go_true_percentage, go_false_percentage, stop_true_percentage, stop_false_percentage]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, color=["green", "red", "blue", "orange"])
    plt.title(f"{method_name} - {dataset_name} Dataset")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

# Paths to CSV files
lime_train_csv = "plots/lime_on_images/lime_masking_train_results.csv"
lime_test_csv = "plots/lime_on_images/lime_masking_test_results.csv"

object_detection_train_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv"
object_detection_test_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"

grid_train_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv"
grid_test_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

# LIME-Based Masking
evaluate_results_with_visualization(lime_train_csv, "Train", "LIME-Based Masking", "plots/lime_train.png")
evaluate_results_with_visualization(lime_test_csv, "Test", "LIME-Based Masking", "plots/lime_test.png")

# Object Detection-Based Masking
evaluate_results_with_visualization(object_detection_train_csv, "Train", "Object Detection-Based Masking", "plots/object_detection_train.png")
evaluate_results_with_visualization(object_detection_test_csv, "Test", "Object Detection-Based Masking", "plots/object_detection_test.png")

# Grid-Based Masking
evaluate_results_with_visualization(grid_train_csv, "Train", "Grid-Based Masking", "plots/grid_train.png")
evaluate_results_with_visualization(grid_test_csv, "Test", "Grid-Based Masking", "plots/grid_test.png")