import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample

def process_and_resample(data_file, output_dir, thresholds):
    # Load the dataset
    df = pd.read_csv(data_file)

    # Define thresholds for labeling
    stop_threshold = thresholds['stop']
    go_threshold = thresholds['go']
    turn_threshold = thresholds['turn']

    print(f"Processing file: {data_file}")
    print("Thresholds:")
    print(f"STOP: brake > {stop_threshold['brake']}, throttle < {stop_threshold['throttle']}")
    print(f"GO: throttle > {go_threshold['throttle']}, abs(steering) < {go_threshold['steering']}")
    print(f"TURN: steering > {turn_threshold['steering']}, brake < {turn_threshold['brake']}")

    # Label data based on thresholds
    def classify_row(row):
        if row['brake'] > stop_threshold['brake'] and row['throttle'] < stop_threshold['throttle']:
            return 'STOP'
        elif row['throttle'] > go_threshold['throttle'] and abs(row['steering']) < go_threshold['steering']:
            return 'GO'
        elif row['steering'] > turn_threshold['steering'] and row['brake'] < turn_threshold['brake']:
            return 'RIGHT'
        elif row['steering'] < -turn_threshold['steering'] and row['brake'] < turn_threshold['brake']:
            return 'LEFT'
        else:
            return 'UNKNOWN'

    df['label'] = df.apply(classify_row, axis=1)

    # Print sample rows
    print("Sample labeled rows:")
    print(df[['steering', 'throttle', 'brake', 'label']].head())

    # Filter out UNKNOWN labels
    initial_count = len(df)
    df = df[df['label'] != 'UNKNOWN']
    filtered_count = len(df)
    print(f"Filtered out {initial_count - filtered_count} UNKNOWN labels")

    # Resample to balance classes
    print("Class distribution before resampling:")
    print(df['label'].value_counts())

    classes = df['label'].unique()
    resampled_dfs = []
    max_class_size = df['label'].value_counts().max()

    for label in classes:
        class_df = df[df['label'] == label]
        if len(class_df) < max_class_size:
            resampled_dfs.append(resample(class_df, replace=True, n_samples=max_class_size, random_state=42))
        else:
            resampled_dfs.append(class_df)

    balanced_df = pd.concat(resampled_dfs)

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the processed and balanced dataset
    os.makedirs(output_dir, exist_ok=True)
    processed_file = os.path.join(output_dir, "processed_balanced_data.csv")
    balanced_df.to_csv(processed_file, index=False)

    print(f"Processed and balanced data saved to {processed_file}")

    # Plot label distribution
    label_counts = balanced_df['label'].value_counts()
    print("Class distribution after resampling:")
    print(label_counts)

    plt.figure(figsize=(8, 8))
    label_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=[0.1]*len(label_counts))
    plt.title(f'Balanced Label Distribution')
    plt.ylabel('')
    plot_file = os.path.join(output_dir, "label_distribution.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Label distribution plot saved to {plot_file}")

# Thresholds based on dataset analysis
thresholds = {
    'stop': {'brake': 0.7, 'throttle': 0.2},
    'go': {'throttle': 0.6, 'steering': 0.05},
    'turn': {'steering': 0.1, 'brake': 0.2}
}

# Process train dataset
process_and_resample(
    data_file="dataset/town7_dataset/train/train_data_log.csv",
    output_dir="plots/dataset_images_for_4_classes/train",
    thresholds=thresholds
)

# Process test dataset
process_and_resample(
    data_file="dataset/town7_dataset/test/test_data_log.csv",
    output_dir="plots/dataset_images_for_4_classes/test",
    thresholds=thresholds
)

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# def process_and_plot(data_file, output_dir):
#     # Load the dataset
#     df = pd.read_csv(data_file)

#     # Calculate statistics for each feature
#     stats = {
#         'steering': df['steering'].describe(),
#         'throttle': df['throttle'].describe(),
#         'brake': df['brake'].describe()
#     }

#     # Add variance to the statistics
#     def calculate_variance(series):
#         return series.var()

#     for feature in ['steering', 'throttle', 'brake']:
#         stats[feature]['variance'] = calculate_variance(df[feature])

#     # Print statistics
#     print(f"Statistics for dataset: {data_file}")
#     for feature, stat in stats.items():
#         print(f"Statistics for {feature}:")
#         print(stat)
#         print()

#     # Identify and remove outliers using IQR
#     for feature in ['steering', 'throttle', 'brake']:
#         Q1 = df[feature].quantile(0.25)
#         Q3 = df[feature].quantile(0.75)
#         IQR = Q3 - Q1
#         outlier_mask = (df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))
#         print(f"Outliers removed for {feature}: {outlier_mask.sum()}")
#         df = df[~outlier_mask]

#     # Normalize features
#     for feature in ['steering', 'throttle', 'brake']:
#         df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

#     # Create output directory for plots
#     os.makedirs(output_dir, exist_ok=True)

#     # Plot distributions
#     for feature in ['steering', 'throttle', 'brake']:
#         plt.figure(figsize=(10, 6))
#         plt.hist(df[feature], bins=50, edgecolor='black', alpha=0.7)
#         plt.title(f"Distribution of {feature} (Normalized)")
#         plt.xlabel(f"{feature} value")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         # Save plot
#         plot_path = os.path.join(output_dir, f"{feature}_distribution.png")
#         plt.savefig(plot_path)
#         plt.close()

#     # Add classification logic
#     def classify_row(row):
#         if row['brake'] > 0.75 and row['throttle'] < 0.2:
#             return 'STOP'
#         elif row['throttle'] > 0.6 and abs(row['steering']) < 0.05:
#             return 'GO'
#         elif row['steering'] > 0.05 and row['brake'] < 0.2:
#             return 'RIGHT'
#         elif row['steering'] < -0.05 and row['brake'] < 0.2:
#             return 'LEFT'
#         else:
#             return 'UNKNOWN'

#     df['label'] = df.apply(classify_row, axis=1)

#     # Save the processed dataset with labels
#     processed_file = os.path.join(output_dir, "processed_data.csv")
#     df.to_csv(processed_file, index=False)

#     print(f"Processed data saved to {processed_file}")

# # Process train dataset
# process_and_plot(
#     data_file="dataset/town7_dataset/train/train_data_log.csv",
#     output_dir="plots/dataset_images_for_4_classes/train"
# )

# # Process test dataset
# process_and_plot(
#     data_file="dataset/town7_dataset/test/test_data_log.csv",
#     output_dir="plots/dataset_images_for_4_classes/test"
# )