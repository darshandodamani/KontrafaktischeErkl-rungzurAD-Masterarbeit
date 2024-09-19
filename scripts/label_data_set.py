import csv
import os

# Threshold values to decide whether to label as STOP or GO
stop_threshold = (
    0.1  # If brake > stop_threshold or throttle < go_threshold, label as STOP
)
go_threshold = 0.2  # If throttle > go_threshold, label as GO

# Paths to  datasets
train_csv = "dataset/town7_dataset/train/train_data_log.csv"
test_csv = "dataset/town7_dataset/test/test_data_log.csv"

# Output paths for labeled data
train_output_csv = "dataset/town7_dataset/train/labeled_train_data_log.csv"
test_output_csv = "dataset/town7_dataset/test/labeled_test_data_log.csv"


def label_data(input_csv, output_csv):
    with open(input_csv, "r") as infile, open(output_csv, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["label"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            throttle = float(row["throttle"])
            brake = float(row["brake"])

            # Label as STOP if brake is applied or throttle is very low
            if brake > stop_threshold or throttle < go_threshold:
                label = "STOP"
            else:
                label = "GO"

            # Write the row with the label to the new CSV file
            row["label"] = label
            writer.writerow({field: row[field] for field in fieldnames})

    print(f"Labeling completed. Labeled data saved to {output_csv}")


# Label the train and test datasets
label_data(train_csv, train_output_csv)
label_data(test_csv, test_output_csv)
