import pandas as pd
import matplotlib.pyplot as plt
import os

def label_dataset(input_csv, output_csv, stop_quantile, go_quantile, turn_quantile, stop_threshold, go_threshold, turn_threshold, threshold_method, plot_path):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Define thresholds based on provided statistics
    stop_brake_threshold = 0.75
    stop_throttle_threshold = 0.25
    go_throttle_threshold = 0.25
    go_steering_threshold = 0.01
    turn_steering_threshold = 0.01
    turn_brake_threshold = 0.75

    print("Thresholds:")
    print(f"STOP: brake > {stop_brake_threshold}, throttle < {stop_throttle_threshold}")
    print(f"GO: throttle > {go_throttle_threshold}, abs(steering) < {go_steering_threshold}")
    print(f"RIGHT: steering > {turn_steering_threshold}, brake < {turn_brake_threshold}")
    print(f"LEFT: steering < {-turn_steering_threshold}, brake < {turn_brake_threshold}")

    # Label data based on thresholds
    def classify_row(row):
        if row['brake'] > stop_brake_threshold and row['throttle'] < stop_throttle_threshold:
            return 'STOP'
        elif row['throttle'] > go_throttle_threshold and abs(row['steering']) < go_steering_threshold:
            return 'GO'
        elif row['steering'] > turn_steering_threshold and row['brake'] < turn_brake_threshold:
            return 'RIGHT'
        elif row['steering'] < -turn_steering_threshold and row['brake'] < turn_brake_threshold:
            return 'LEFT'
        else:
            return 'UNKNOWN'

    df['label'] = df.apply(classify_row, axis=1)

    # Print the first few labeled rows
    print("Sample labeled rows:")
    print(df[['steering', 'throttle', 'brake', 'label']].head())

    # Filter out UNKNOWN labels
    df = df[df['label'] != 'UNKNOWN']

    # Save the labeled dataset
    df.to_csv(output_csv, index=False)
    print(f"Labeled dataset saved to {output_csv}")

    # Plot label distribution
    label_counts = df['label'].value_counts()
    print("Label distribution:")
    print(label_counts)

    plt.figure(figsize=(8, 8))
    label_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=[0.1] * len(label_counts))
    plt.title(f'Label Distribution')
    plt.ylabel('')
    plt.savefig(plot_path)
    plt.close()
    print(f"Label distribution plot saved to {plot_path}")

# Train dataset processing
label_dataset(
    input_csv=os.path.join("dataset/town7_dataset/train", "train_data_log.csv"),
    output_csv=os.path.join("dataset/town7_dataset/train", "labeled_train_4_class_data_log.csv"),
    stop_quantile=0.75,
    go_quantile=0.25,
    turn_quantile=0.75,
    stop_threshold=0.75,
    go_threshold=0.25,
    turn_threshold=0.01,
    threshold_method="manual",
    plot_path=os.path.join("plots/dataset_images_for_4_classes", "train_label_distribution.png")
)

# Test dataset processing
label_dataset(
    input_csv=os.path.join("dataset/town7_dataset/test", "test_data_log.csv"),
    output_csv=os.path.join("dataset/town7_dataset/test", "labeled_test_4_class_data_log.csv"),
    stop_quantile=0.75,
    go_quantile=0.25,
    turn_quantile=0.75,
    stop_threshold=0.75,
    go_threshold=0.25,
    turn_threshold=0.01,
    threshold_method="manual",
    plot_path=os.path.join("plots/dataset_images_for_4_classes", "test_label_distribution.png")
)
