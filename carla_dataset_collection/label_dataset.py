import pandas as pd
import argparse
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.info(f"Current working directory: {os.getcwd()}")

def label_dataset(input_csv, output_csv, stop_quantile=0.9, go_quantile=0.1, turn_quantile=0.9,
                  stop_threshold=0.1, go_threshold=0.5, turn_threshold=0.1,
                  threshold_method='quantile', plot_path=None):
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

    # Labeling logic for 4 classes (no SKIP)
    def label_row(row):
        if row['brake'] > stop_threshold:  # STOP has the highest priority
            return 'STOP'
        elif row['steering'] > turn_threshold and row['throttle'] > 0.1:  # RIGHT
            return 'RIGHT'
        elif row['steering'] < -turn_threshold and row['throttle'] > 0.1:  # LEFT
            return 'LEFT'
        else:  # Default to GO
            return 'GO'

    # Apply labeling logic
    df['label'] = df.apply(label_row, axis=1)

    # Update counts
    stop_count = df[df['label'] == 'STOP'].shape[0]
    go_count = df[df['label'] == 'GO'].shape[0]
    right_count = df[df['label'] == 'RIGHT'].shape[0]
    left_count = df[df['label'] == 'LEFT'].shape[0]

    logging.info(f"Initial counts - STOP: {stop_count}, GO: {go_count}, RIGHT: {right_count}, LEFT: {left_count}")

    # Total dataset size
    total_size = len(df)
    stop_ratio, go_ratio, right_ratio, left_ratio = 0.35, 0.35, 0.15, 0.15
    target_stop = int(total_size * stop_ratio)
    target_go = int(total_size * go_ratio)
    target_right = int(total_size * right_ratio)
    target_left = int(total_size * left_ratio)

    logging.info(f"Target count per class - STOP: {target_stop}, GO: {target_go}, RIGHT: {target_right}, LEFT: {target_left}")

    # Proportional sampling
    stop_df = df[df['label'] == 'STOP'].sample(n=min(target_stop, stop_count), random_state=42)
    go_df = df[df['label'] == 'GO'].sample(n=min(target_go, go_count), random_state=42)
    right_df = df[df['label'] == 'RIGHT'].sample(n=target_right, replace=True, random_state=42) if len(df[df['label'] == 'RIGHT']) < target_right else df[df['label'] == 'RIGHT'].sample(n=target_right, random_state=42)
    left_df = df[df['label'] == 'LEFT'].sample(n=target_left, replace=True, random_state=42) if len(df[df['label'] == 'LEFT']) < target_left else df[df['label'] == 'LEFT'].sample(n=target_left, random_state=42)

    # Combine balanced data
    balanced_df = pd.concat([stop_df, go_df, right_df, left_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the balanced dataset
    balanced_df.to_csv(output_csv, index=False)

    # Final counts
    final_stop_count = balanced_df[balanced_df['label'] == 'STOP'].shape[0]
    final_go_count = balanced_df[balanced_df['label'] == 'GO'].shape[0]
    final_right_count = balanced_df[balanced_df['label'] == 'RIGHT'].shape[0]
    final_left_count = balanced_df[balanced_df['label'] == 'LEFT'].shape[0]

    logging.info(f"Final counts - STOP: {final_stop_count}, GO: {final_go_count}, RIGHT: {final_right_count}, LEFT: {final_left_count}")
    logging.info(f"Labeled dataset saved to {output_csv}")

    # Plot distributions
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.pie(
            [final_stop_count, final_go_count, final_right_count, final_left_count],
            labels=['STOP', 'GO', 'RIGHT', 'LEFT'],
            colors=['red', 'green', 'blue', 'yellow'],
            autopct='%1.1f%%'
        )
        plt.title(f'Label Distribution: {os.path.basename(output_csv)}')
        plt.savefig(plot_path)
        logging.info(f"Plot saved to {plot_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Labeling Script for Train and Test Datasets")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the train dataset.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--threshold_method", type=str, choices=['fixed', 'quantile'], default='quantile', help="Method to determine thresholds.")
    parser.add_argument("--stop_quantile", type=float, default=0.9, help="Quantile for STOP threshold (quantile method).")
    parser.add_argument("--go_quantile", type=float, default=0.1, help="Quantile for GO threshold (quantile method).")
    parser.add_argument("--turn_quantile", type=float, default=0.9, help="Quantile for TURN threshold (quantile method).")
    parser.add_argument("--stop_threshold", type=float, default=0.1, help="Fixed STOP threshold (fixed method).")
    parser.add_argument("--go_threshold", type=float, default=0.5, help="Fixed GO threshold (fixed method).")
    parser.add_argument("--turn_threshold", type=float, default=0.1, help="Fixed TURN threshold (fixed method).")
    parser.add_argument("--plot_output_path", type=str, default='./plots/dataset_images_for_4_classes', help="Base path to save plots.")
    args = parser.parse_args()

    # Train dataset processing
    label_dataset(
        input_csv=os.path.join(args.train_data_path, "train_data_log.csv"),
        output_csv=os.path.join(args.train_data_path, "labeled_train_4_class_data_log.csv"),
        stop_quantile=args.stop_quantile,
        go_quantile=args.go_quantile,
        turn_quantile=args.turn_quantile,
        stop_threshold=args.stop_threshold,
        go_threshold=args.go_threshold,
        turn_threshold=args.turn_threshold,
        threshold_method=args.threshold_method,
        plot_path=os.path.join(args.plot_output_path, "train_label_distribution.png")
    )

    # Test dataset processing
    label_dataset(
        input_csv=os.path.join(args.test_data_path, "test_data_log.csv"),
        output_csv=os.path.join(args.test_data_path, "labeled_test_4_class_data_log.csv"),
        stop_quantile=args.stop_quantile,
        go_quantile=args.go_quantile,
        turn_quantile=args.turn_quantile,
        stop_threshold=args.stop_threshold,
        go_threshold=args.go_threshold,
        turn_threshold=args.turn_threshold,
        threshold_method=args.threshold_method,
        plot_path=os.path.join(args.plot_output_path, "test_label_distribution.png")
    )
