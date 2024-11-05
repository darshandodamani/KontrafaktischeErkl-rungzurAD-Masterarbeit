# label_dataset.py
import pandas as pd
import argparse
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def label_dataset(data_path, stop_threshold=0.1, go_threshold=0.2, stop_quantile=0.9, go_quantile=0.1,
                  threshold_method='quantile', balance_method='oversample', plot_output_path='../plots/dataset_images'):
    input_csv = os.path.join(data_path, "Town05_data_log.csv")
    output_csv = os.path.join(data_path, "labeled_data_log.csv")
    df = pd.read_csv(input_csv)

    # Initialize counters for transparency
    total_records = len(df)
    total_labeled = 0
    stop_count = 0
    go_count = 0
    skipped_records = 0

    # Determine thresholds based on the chosen method
    if threshold_method == 'quantile':
        stop_threshold = df['brake'].quantile(stop_quantile)
        go_threshold = df['throttle'].quantile(go_quantile)
        logging.info(f"Using dynamic thresholds based on quantiles: STOP (brake quantile {stop_quantile}): {stop_threshold}, GO (throttle quantile {go_quantile}): {go_threshold}")
    elif threshold_method == 'fixed':
        logging.info(f"Using fixed thresholds: STOP: {stop_threshold}, GO: {go_threshold}")
    else:
        logging.error("Unknown threshold method. Use 'fixed' or 'quantile'.")
        return

    # Labeling logic
    df['label'] = df.apply(lambda row: 'STOP' if row['brake'] > stop_threshold or row['throttle'] < go_threshold else 'GO', axis=1)

    # Update counters
    stop_count = df[df['label'] == 'STOP'].shape[0]
    go_count = df[df['label'] == 'GO'].shape[0]
    total_labeled = stop_count + go_count
    skipped_records = total_records - total_labeled

    # Balance STOP and GO labels
    stop_df = df[df['label'] == 'STOP']
    go_df = df[df['label'] == 'GO']

    oversampled_count = 0
    if len(stop_df) == 0 or len(go_df) == 0:
        logging.warning("One of the classes has zero samples. Skipping balancing.")
        balanced_df = df
    else:
        if balance_method == 'oversample':
            if len(stop_df) < len(go_df):
                oversampled_count = len(go_df) - len(stop_df)
                stop_df = stop_df.sample(len(go_df), replace=True, random_state=42)
                logging.info(f"Oversampled STOP class by adding {oversampled_count} duplicated records to match GO class.")
            elif len(go_df) < len(stop_df):
                oversampled_count = len(stop_df) - len(go_df)
                go_df = go_df.sample(len(stop_df), replace=True, random_state=42)
                logging.info(f"Oversampled GO class by adding {oversampled_count} duplicated records to match STOP class.")
        elif balance_method == 'undersample':
            min_count = min(len(stop_df), len(go_df))
            stop_df = stop_df.sample(min_count, random_state=42)
            go_df = go_df.sample(min_count, random_state=42)
            logging.info(f"Undersampled both classes to {min_count} samples each.")
        else:
            logging.warning("Unknown balance method, proceeding without balancing.")

        balanced_df = pd.concat([stop_df, go_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_df.to_csv(output_csv, index=False)

    # Log final counts
    stop_count_balanced = balanced_df[balanced_df['label'] == 'STOP'].shape[0]
    go_count_balanced = balanced_df[balanced_df['label'] == 'GO'].shape[0]
    total_balanced = len(balanced_df)
    duplicate_rate = (oversampled_count / total_balanced) * 100 if total_balanced > 0 else 0

    logging.info(f"Labeling completed. STOP: {stop_count_balanced}, GO: {go_count_balanced}")
    logging.info(f"Total records processed: {total_records}")
    logging.info(f"Total labeled records: {total_labeled}")
    logging.info(f"Total skipped records: {skipped_records}")
    logging.info(f"Total records after balancing: {total_balanced} (including {oversampled_count} duplicated records from oversampling)")
    logging.info(f"Percentage of duplicated records due to oversampling: {duplicate_rate:.2f}%")

    # Plotting the brake and throttle distributions to assist threshold setting
    plt.figure(figsize=(10, 5))
    plt.hist(df['brake'], bins=50, alpha=0.5, label='Brake', color='red')
    plt.hist(df['throttle'], bins=50, alpha=0.5, label='Throttle', color='green')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Brake and Throttle')
    plt.legend()
    plot_dist_path = os.path.join(plot_output_path, 'brake_throttle_distribution.png')
    os.makedirs(plot_output_path, exist_ok=True)
    plt.savefig(plot_dist_path)
    logging.info(f"Brake and throttle distribution plot saved to {plot_dist_path}")

    # Plotting the labeled dataset as a pie chart
    plt.figure(figsize=(8, 5))
    plt.pie([stop_count_balanced, go_count_balanced], labels=['STOP', 'GO'], colors=['red', 'green'], autopct='%1.1f%%')
    plt.title('Distribution of STOP and GO Labels')
    plot_path = os.path.join(plot_output_path, 'label_distribution.png')
    plt.savefig(plot_path)
    logging.info(f"Label distribution plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Labeling Script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the collected dataset.")
    parser.add_argument("--threshold_method", type=str, choices=['fixed', 'quantile'], default='quantile', help="Method to determine thresholds ('fixed' or 'quantile').")
    parser.add_argument("--stop_threshold", type=float, default=0.1, help="Fixed threshold to determine STOP label (only used if threshold_method is 'fixed').")
    parser.add_argument("--go_threshold", type=float, default=0.2, help="Fixed threshold to determine GO label (only used if threshold_method is 'fixed').")
    parser.add_argument("--stop_quantile", type=float, default=0.9, help="Quantile to determine STOP threshold (only used if threshold_method is 'quantile').")
    parser.add_argument("--go_quantile", type=float, default=0.1, help="Quantile to determine GO threshold (only used if threshold_method is 'quantile').")
    parser.add_argument("--balance_method", type=str, choices=['oversample', 'undersample'], default='oversample', help="Method to balance STOP and GO labels ('oversample' or 'undersample').")
    parser.add_argument("--plot_output_path", type=str, default='../plots/dataset_images', help="Path to save plots.")
    args = parser.parse_args()

    label_dataset(
        data_path=args.data_path,
        stop_threshold=args.stop_threshold,
        go_threshold=args.go_threshold,
        stop_quantile=args.stop_quantile,
        go_quantile=args.go_quantile,
        threshold_method=args.threshold_method,
        balance_method=args.balance_method,
        plot_output_path=args.plot_output_path
    )
