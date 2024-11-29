import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_and_plot_metrics(csv_file, output_dir):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Metrics to plot
    metrics = ["SSIM", "MSE", "PSNR", "UQI", "VIFP"]
    
    # Iterate over the metrics and plot their distributions
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(df[metric], bins=20, color='b', alpha=0.7)
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {metric} values')
        plt.grid(axis='y', linestyle='--')
        output_file = os.path.join(output_dir, f"{metric}_distribution.png")
        plt.savefig(output_file)
        plt.close()

    # Plot comparison of metrics for all images
    df_metrics = df[metrics]
    df_metrics_mean = df_metrics.mean()

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x, df_metrics_mean, width, color='g', alpha=0.6)
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('Average Metric Values for Counterfactual Explanations')
    plt.xticks(x, metrics)
    plt.grid(axis='y', linestyle='--')
    output_file = os.path.join(output_dir, "average_metrics_values.png")
    plt.savefig(output_file)
    plt.close()

    # Count the number of counterfactual explanations generated for STOP and GO
    stop_count = df[df['Prediction'] == 'STOP']['Counterfactual Found'].sum()
    go_count = df[df['Prediction'] == 'GO']['Counterfactual Found'].sum()

    # Plot the count of counterfactual explanations for STOP and GO
    plt.figure(figsize=(10, 6))
    labels = ['STOP', 'GO']
    counts = [stop_count, go_count]
    plt.bar(labels, counts, color=['r', 'b'], alpha=0.7)
    plt.xlabel('Prediction')
    plt.ylabel('Number of Counterfactual Explanations')
    plt.title('Number of Counterfactual Explanations for STOP and GO')
    plt.grid(axis='y', linestyle='--')
    output_file = os.path.join(output_dir, "counterfactual_counts.png")
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    # Define paths to CSV files
    train_csv = 'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv'
    test_csv = 'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv'

    # Output directory for plots
    output_dir = 'plots/evaluation_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Plot metrics for train dataset
    read_and_plot_metrics(train_csv, output_dir)

    # Plot metrics for test dataset
    read_and_plot_metrics(test_csv, output_dir)
