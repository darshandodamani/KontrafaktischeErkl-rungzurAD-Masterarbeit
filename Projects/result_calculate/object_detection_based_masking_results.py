import pandas as pd

# Load train and test CSV files
train_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv"
test_csv = "plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv"

# Load data
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Combine train and test data
data = pd.concat([train_data, test_data], ignore_index=True)

# Initialize metrics dictionary
metrics = []

# Total entries and total time
total_time = data["Processing Time (s)"].sum()
total_entries = len(data)

# Initialize overall counters
overall_ce_found = 0
overall_ce_not_found = 0

# Process each prediction class (GO and STOP)
for prediction_class in ["GO", "STOP"]:
    # Filter data by the initial prediction class
    class_data = data[data["Initial Prediction"] == prediction_class]
    
    # Total cases for this class
    total_cases = class_data.shape[0]
    
    # Counterfactual Found and Not Found
    ce_found_count = class_data[class_data["Counterfactual Found"] == True].shape[0]
    ce_not_found_count = class_data[class_data["Counterfactual Found"] == False].shape[0]
    
    # Percentages
    ce_found_percentage = (ce_found_count / total_cases) * 100 if total_cases > 0 else 0
    ce_not_found_percentage = (ce_not_found_count / total_cases) * 100 if total_cases > 0 else 0

    # Add to overall counters
    overall_ce_found += ce_found_count
    overall_ce_not_found += ce_not_found_count

    # Add Total GO/STOP row
    metrics.append({
        "Metrics": f"Total {prediction_class}",
        "Total Count": total_cases,
        "Count": "",
        "Percentage": ""
    })
    
    # Add Counterfactual Found row
    metrics.append({
        "Metrics": f"{prediction_class} (Counterfactual Found)",
        "Total Count": "",
        "Count": ce_found_count,
        "Percentage": f"{ce_found_percentage:.2f}%"
    })
    
    # Add Counterfactual Not Found row
    metrics.append({
        "Metrics": f"{prediction_class} (Counterfactual Not Found)",
        "Total Count": "",
        "Count": ce_not_found_count,
        "Percentage": f"{ce_not_found_percentage:.2f}%"
    })

# Add Overall Total and Percentages
overall_found_percentage = (overall_ce_found / total_entries) * 100 if total_entries > 0 else 0
overall_not_found_percentage = (overall_ce_not_found / total_entries) * 100 if total_entries > 0 else 0

metrics.append({
    "Metrics": "Total",
    "Total Count": total_entries,
    "Count": "",
    "Percentage": ""
})
metrics.append({
    "Metrics": "Total (Counterfactual Found)",
    "Total Count": "",
    "Count": overall_ce_found,
    "Percentage": f"{overall_found_percentage:.2f}%"
})
metrics.append({
    "Metrics": "Total (Counterfactual Not Found)",
    "Total Count": "",
    "Count": overall_ce_not_found,
    "Percentage": f"{overall_not_found_percentage:.2f}%"
})
metrics.append({
    "Metrics": "Total Time Taken",
    "Total Count": "",
    "Count": f"{total_time:.2f} seconds",
    "Percentage": ""
})

# Convert metrics to a DataFrame
summary_table = pd.DataFrame(metrics)

# Save to CSV
output_file = "plots/object_detection_using_yolov5/object_detection_summary.csv"
summary_table.to_csv(output_file, index=False)

# Print results to terminal
print("\nObject Detection Masking Results:")
print(summary_table)
