import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
grid_based_test = pd.read_csv('/home/selab/darshan/git-repos/plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv')
object_detection_test = pd.read_csv('/home/selab/darshan/git-repos/plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv')

# Filter rows where 'Counterfactual Found' is True (assuming it's a boolean or 1/0 column)
grid_based_filtered = grid_based_test.loc[grid_based_test['Counterfactual Found'] == True].copy()
object_detection_filtered = object_detection_test.loc[object_detection_test['Counterfactual Found'] == True].copy()

# Clean and standardize 'Image File' columns
grid_based_filtered['Image File'] = grid_based_filtered['Image File'].str.strip().str.lower()
object_detection_filtered['Image File'] = object_detection_filtered['Image File'].str.strip().str.lower()

# Function to check if two ranges overlap
def is_range_overlap(grid_position, coordinate_range_str):
    try:
        # Convert grid_position to a range
        grid_position = float(grid_position)
        grid_range = (grid_position, grid_position)
        
        # Extract coordinate pairs from the coordinate range string
        coordinate_pairs = re.findall(r'\((\d+), (\d+)\)', coordinate_range_str)
        if not coordinate_pairs:
            return False
        
        for pair in coordinate_pairs:
            x1, y1 = map(int, pair)
            coord_range = (min(x1, y1), max(x1, y1))
            
            # Check if the ranges overlap
            if (grid_range[0] <= coord_range[1] and grid_range[1] >= coord_range[0]):
                return True
        return False
    except:
        return False

# Perform matching based on Image File, Grid Position, and Prediction
matched_rows = []
for _, grid_row in grid_based_filtered.iterrows():
    matching_rows = object_detection_filtered[object_detection_filtered['Image File'] == grid_row['Image File']]
    for _, obj_row in matching_rows.iterrows():
        if (is_range_overlap(grid_row['Grid Position'], obj_row['Grid Position']) and
                grid_row['Prediction'] == obj_row['Prediction']):
            matched_rows.append({**grid_row, **obj_row})

# Convert the matched rows to a DataFrame
matched_counterfactuals = pd.DataFrame(matched_rows)

# Display the matched results
print("Matched Counterfactuals with Overlapping Grid Position and Same Prediction:")
print(matched_counterfactuals)

# Save the matched results to a CSV file (optional)
matched_counterfactuals.to_csv('plots/comparison_plots/matched_counterfactuals.csv', index=False)

# Debugging output: Print a few rows from both filtered datasets
print("\nGrid-Based Filtered Sample:")
print(grid_based_filtered.head())
print("\nObject Detection Filtered Sample:")
print(object_detection_filtered.head())

# Visualization: Plot the number of matches per image
if not matched_counterfactuals.empty:
    match_counts = matched_counterfactuals['Image File'].value_counts()
    
    plt.figure(figsize=(10, 6))
    match_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Image File')
    plt.ylabel('Number of Matches')
    plt.title('Number of Matched Counterfactuals per Image')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/comparison_plots/matches_per_image.png')
    plt.close()

    # Statistical Analysis: Descriptive statistics of the matched counterfactuals
    descriptive_stats = matched_counterfactuals[['Confidence', 'SSIM', 'MSE', 'PSNR', 'UQI', 'VIFP']].describe()
    print("\nDescriptive Statistics of Matched Counterfactuals:")
    print(descriptive_stats)

    # Grouped Analysis: Descriptive statistics per Prediction type
    grouped_stats = matched_counterfactuals.groupby('Prediction')[['Confidence', 'SSIM', 'MSE', 'PSNR', 'UQI', 'VIFP']].describe()
    print("\nGrouped Descriptive Statistics by Prediction Type:")
    print(grouped_stats)

    # Correlation Analysis: Correlation between confidence and quality metrics
    correlation_matrix = matched_counterfactuals[['Confidence', 'SSIM', 'MSE', 'PSNR', 'UQI', 'VIFP']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('Correlation Matrix of Quality Metrics')
    plt.savefig('plots/comparison_plots/correlation_matrix.png')
    plt.close()

    # Scatter plots for metric comparisons
    plt.figure(figsize=(10, 6))
    plt.scatter(matched_counterfactuals['Confidence'], matched_counterfactuals['SSIM'], alpha=0.6)
    plt.xlabel('Confidence')
    plt.ylabel('SSIM')
    plt.title('Confidence vs SSIM')
    plt.savefig('plots/comparison_plots/confidence_vs_ssim.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(matched_counterfactuals['MSE'], matched_counterfactuals['PSNR'], alpha=0.6, color='orange')
    plt.xlabel('MSE')
    plt.ylabel('PSNR')
    plt.title('MSE vs PSNR')
    plt.savefig('plots/comparison_plots/mse_vs_psnr.png')
    plt.close()

    # Pair plot for visualizing relationships between all quality metrics
    sns.pairplot(matched_counterfactuals[['Confidence', 'SSIM', 'MSE', 'PSNR', 'UQI', 'VIFP']])
    plt.savefig('plots/comparison_plots/pairplot_quality_metrics.png')
    plt.close()

    # Additional line plot for changes in metrics across matched images (if applicable)
    plt.figure(figsize=(12, 6))
    for metric in ['Confidence', 'SSIM', 'MSE', 'PSNR', 'UQI', 'VIFP']:
        plt.plot(matched_counterfactuals.index, matched_counterfactuals[metric], label=metric)
    plt.xlabel('Matched Instances Index')
    plt.ylabel('Metric Value')
    plt.title('Changes in Metrics Across Matched Instances')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/comparison_plots/metrics_over_time.png')
    plt.close()
