# label_dataset.py
import pandas as pd
import argparse
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def label_dataset(data_path, stop_threshold=0.1, go_threshold=0.2):
    input_csv = os.path.join(data_path, "Town06_data_log.csv")
    output_csv = os.path.join(data_path, "labeled_data_log.csv")
    df = pd.read_csv(input_csv)
    df['label'] = df.apply(lambda row: 'STOP' if row['brake'] > stop_threshold or row['throttle'] < go_threshold else 'GO', axis=1)

    # Balance STOP and GO labels by oversampling the minority class
    stop_df = df[df['label'] == 'STOP']
    go_df = df[df['label'] == 'GO']
    if len(stop_df) < len(go_df):
        stop_df = stop_df.sample(len(go_df), replace=True, random_state=42)
    elif len(go_df) < len(stop_df):
        go_df = go_df.sample(len(stop_df), replace=True, random_state=42)

    balanced_df = pd.concat([stop_df, go_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(output_csv, index=False)

    stop_count = balanced_df[balanced_df['label'] == 'STOP'].shape[0]
    go_count = balanced_df[balanced_df['label'] == 'GO'].shape[0]
    logging.info(f"Labeling completed. STOP: {stop_count}, GO: {go_count}")

    # Validate that the total number of labeled entries matches the original count
    total_labeled = stop_count + go_count
    assert len(balanced_df) == total_labeled, "The total number of labeled entries does not match the original count."
    logging.info("Label validation successful: Total entries match after labeling.")

    # Plotting the labeled dataset as a pie chart
    os.makedirs(os.path.join('..', 'plots', 'dataset_images'), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.pie([stop_count, go_count], labels=['STOP', 'GO'], colors=['red', 'green'], autopct='%1.1f%%')
    plt.title('Distribution of STOP and GO Labels')
    plot_path = os.path.join('..', 'plots', 'dataset_images', 'label_distribution.png')
    plt.savefig(plot_path)
    logging.info(f"Label distribution plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Labeling Script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the collected dataset.")
    parser.add_argument("--stop_threshold", type=float, default=0.1, help="Threshold to determine STOP label.")
    parser.add_argument("--go_threshold", type=float, default=0.2, help="Threshold to determine GO label.")
    args = parser.parse_args()

    label_dataset(args.data_path, args.stop_threshold, args.go_threshold)