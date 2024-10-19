# split_dataset.py
import os
import pandas as pd
import shutil
import argparse
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def split_dataset(data_path, train_ratio=0.8):
    input_csv = os.path.join(data_path, "labeled_data_log.csv")
    train_folder = os.path.join(data_path, "train")
    test_folder = os.path.join(data_path, "test")
    train_csv = os.path.join(train_folder, "train_data_log.csv")
    test_csv = os.path.join(test_folder, "test_data_log.csv")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    data_df = pd.read_csv(input_csv)
    total_images = len(data_df)
    logging.info(f"Total images: {total_images}")

    # Ensure balanced STOP and GO labels in both train and test sets
    stop_df = data_df[data_df['label'] == 'STOP']
    go_df = data_df[data_df['label'] == 'GO']
    if len(stop_df) < len(go_df):
        stop_df = stop_df.sample(len(go_df), replace=True, random_state=42)
    elif len(go_df) < len(stop_df):
        go_df = go_df.sample(len(stop_df), replace=True, random_state=42)

    balanced_df = pd.concat([stop_df, go_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    total_balanced_images = len(balanced_df)
    logging.info(f"Total balanced images: {total_balanced_images}")

    # Splitting the balanced dataset
    train_df = balanced_df.sample(frac=train_ratio, random_state=42)
    test_df = balanced_df.drop(train_df.index)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Log the number of images in train and test sets
    logging.info(f"Dataset split completed. Train set: {len(train_df)}, Test set: {len(test_df)}")

    # Copy files to respective folders
    def copy_files(file_list, source_folder, dest_folder):
        for filename in file_list:
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)

    copy_files(train_df['image_filename'], data_path, train_folder)
    copy_files(test_df['image_filename'], data_path, test_folder)

    # Validate the split
    total_split_images = len(train_df) + len(test_df)
    assert total_balanced_images == total_split_images, "The total number of images after splitting does not match the original count."
    logging.info("Split validation successful: Total images match after splitting.")

    # Plotting the dataset split as a pie chart
    os.makedirs(os.path.join('..', 'plots', 'dataset_images'), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.pie([len(train_df), len(test_df)], labels=['Train', 'Test'], colors=['blue', 'orange'], autopct='%1.1f%%')
    plt.title('Train/Test Split of Dataset')
    plot_path = os.path.join('..', 'plots', 'dataset_images', 'dataset_split.png')
    plt.savefig(plot_path)
    logging.info(f"Dataset split plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Splitting Script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the collected dataset.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training.")
    args = parser.parse_args()

    split_dataset(args.data_path, args.train_ratio)