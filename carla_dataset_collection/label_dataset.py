import pandas as pd
import argparse
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def label_dataset(data_path, stop_quantile=0.9, go_quantile=0.1, turn_quantile=0.9, 
                  stop_threshold=0.1, go_threshold=0.5, turn_threshold=0.1, 
                  threshold_method='quantile', balance_method='oversample', 
                  plot_output_path='../plots/dataset_for_4_classes'):
    input_csv = os.path.join(data_path, "test_data_log.csv")
    output_csv = os.path.join(data_path, "labeled_4_classes_data_log.csv")
    df = pd.read_csv(input_csv)

    # Determine thresholds based on the chosen method
    if threshold_method == 'quantile':
        stop_threshold = df['brake'].quantile(stop_quantile)
        go_threshold = df['throttle'].quantile(go_quantile)
        turn_threshold = df['steering'].quantile(turn_quantile)
        logging.info(f"Using dynamic thresholds based on quantiles: STOP (brake): {stop_threshold}, GO (throttle): {go_threshold}, TURN (steering): {turn_threshold}")
    elif threshold_method == 'fixed':
        logging.info(f"Using fixed thresholds: STOP: {stop_threshold}, GO: {go_threshold}, TURN: {turn_threshold}")
    else:
        logging.error("Unknown threshold method. Use 'fixed' or 'quantile'.")
        return

    # Labeling logic for 4 classes
    def label_row(row):
        if row['brake'] > stop_threshold:  # STOP
            return 'STOP'
        elif row['throttle'] > go_threshold:  # GO
            return 'GO'
        elif row['steering'] > turn_threshold:  # RIGHT
            return 'RIGHT'
        elif row['steering'] < -turn_threshold:  # LEFT
            return 'LEFT'
        else:
            return 'SKIP'

    # Apply labeling logic
    df['label'] = df.apply(label_row, axis=1)

    # Filter out skipped records
    df = df[df['label'] != 'SKIP']

    # Update counts
    stop_count = df[df['label'] == 'STOP'].shape[0]
    go_count = df[df['label'] == 'GO'].shape[0]
    right_count = df[df['label'] == 'RIGHT'].shape[0]
    left_count = df[df['label'] == 'LEFT'].shape[0]

    logging.info(f"Initial counts - STOP: {stop_count}, GO: {go_count}, RIGHT: {right_count}, LEFT: {left_count}")

    # Balance the dataset
    stop_df = df[df['label'] == 'STOP']
    go_df = df[df['label'] == 'GO']
    right_df = df[df['label'] == 'RIGHT']
    left_df = df[df['label'] == 'LEFT']

    if balance_method == 'oversample':
        max_count = max(stop_count, go_count, right_count, left_count)
        stop_df = stop_df.sample(max_count, replace=True, random_state=42) if len(stop_df) > 0 else pd.DataFrame()
        go_df = go_df.sample(max_count, replace=True, random_state=42) if len(go_df) > 0 else pd.DataFrame()
        right_df = right_df.sample(max_count, replace=True, random_state=42) if len(right_df) > 0 else pd.DataFrame()
        left_df = left_df.sample(max_count, replace=True, random_state=42) if len(left_df) > 0 else pd.DataFrame()
    elif balance_method == 'undersample':
        min_count = min(stop_count, go_count, right_count, left_count)
        stop_df = stop_df.sample(min_count, random_state=42)
        go_df = go_df.sample(min_count, random_state=42)
        right_df = right_df.sample(min_count, random_state=42)
        left_df = left_df.sample(min_count, random_state=42)

    # Combine balanced data
    balanced_df = pd.concat([stop_df, go_df, right_df, left_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(output_csv, index=False)

    # Final counts
    logging.info(f"Balanced counts - STOP: {len(stop_df)}, GO: {len(go_df)}, RIGHT: {len(right_df)}, LEFT: {len(left_df)}")
    logging.info(f"Labeled dataset saved to {output_csv}")

    # Plot distributions
    os.makedirs(plot_output_path, exist_ok=True)

    # Brake, throttle, and steering distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df['brake'], bins=50, alpha=0.5, label='Brake', color='red')
    plt.hist(df['throttle'], bins=50, alpha=0.5, label='Throttle', color='green')
    plt.hist(df['steering'], bins=50, alpha=0.5, label='Steering', color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distributions of Brake, Throttle, and Steering')
    plt.legend()
    plt.savefig(os.path.join(plot_output_path, 'brake_throttle_steering_distribution.png'))
    plt.close()

    # Label distribution
    plt.figure(figsize=(8, 5))
    plt.pie(
        [len(stop_df), len(go_df), len(right_df), len(left_df)],
        labels=['STOP', 'GO', 'RIGHT', 'LEFT'],
        colors=['red', 'green', 'blue', 'yellow'],
        autopct='%1.1f%%'
    )
    plt.title('Label Distribution')
    plt.savefig(os.path.join(plot_output_path, 'label_distribution.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Labeling Script for 4 Classes")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--threshold_method", type=str, choices=['fixed', 'quantile'], default='quantile', help="Method to determine thresholds.")
    parser.add_argument("--stop_quantile", type=float, default=0.9, help="Quantile for STOP threshold (quantile method).")
    parser.add_argument("--go_quantile", type=float, default=0.1, help="Quantile for GO threshold (quantile method).")
    parser.add_argument("--turn_quantile", type=float, default=0.9, help="Quantile for TURN threshold (quantile method).")
    parser.add_argument("--stop_threshold", type=float, default=0.1, help="Fixed STOP threshold (fixed method).")
    parser.add_argument("--go_threshold", type=float, default=0.5, help="Fixed GO threshold (fixed method).")
    parser.add_argument("--turn_threshold", type=float, default=0.1, help="Fixed TURN threshold (fixed method).")
    parser.add_argument("--balance_method", type=str, choices=['oversample', 'undersample'], default='oversample', help="Balancing method for labels.")
    parser.add_argument("--plot_output_path", type=str, default='../plots/dataset_for_4_classes', help="Path to save plots.")
    args = parser.parse_args()

    label_dataset(
        data_path=args.data_path,
        stop_quantile=args.stop_quantile,
        go_quantile=args.go_quantile,
        turn_quantile=args.turn_quantile,
        stop_threshold=args.stop_threshold,
        go_threshold=args.go_threshold,
        turn_threshold=args.turn_threshold,
        threshold_method=args.threshold_method,
        balance_method=args.balance_method,
        plot_output_path=args.plot_output_path
    )
