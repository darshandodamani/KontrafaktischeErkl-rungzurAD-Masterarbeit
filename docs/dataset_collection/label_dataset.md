# location: docs/dataset_collection/lable_dataset
# Labeling the Dataset
The dataset is labeled based on the throttle and brake values collected from the vehicle in CARLA. Images are labeled as STOP or GO based on thresholds for brake and throttle, which can be determined either using fixed values or dynamically based on quantiles:

### Threshold Methods:
- **Fixed Thresholds**: You can set specific threshold values for labeling.
  - **STOP**: If brake > `stop_threshold` or throttle < `go_threshold`.
  - **GO**: Otherwise.
- **Quantile Thresholds**: Thresholds are determined dynamically based on the quantile of the brake and throttle values in the dataset.
  - **STOP**: If brake > `stop_quantile` or throttle < `go_quantile`.
  - **GO**: Otherwise.

### Justification for Threshold Selection:
The values for the fixed thresholds (`brake > 0.15` or `throttle < 0.25`) were chosen based on an analysis of the distribution of brake and throttle data collected from the vehicle in CARLA:
- **Brake Threshold (`0.15`)**: This value was selected because it effectively separates scenarios where the vehicle is braking significantly, as indicated by the data distribution. A value greater than `0.15` generally implies that the brake is being applied to slow down or stop.
- **Throttle Threshold (`0.25`)**: This value was chosen because it differentiates between low-throttle scenarios, where the vehicle is likely to be idling or moving very slowly, and higher-throttle scenarios, where the vehicle is actively accelerating.
- The distribution plot of brake and throttle values (shown in the `brake_throttle_distribution.png` file) indicated that these thresholds provide a balanced division between `STOP` and `GO` scenarios, resulting in equal representation of both classes in the dataset (50% `STOP` and 50% `GO`).

### Command to Label the Dataset:
To label the dataset, use the following command:

```bash
python label_dataset.py --data_path <path_to_dataset>
```

### Example Command:
```bash
python label_dataset.py --data_path ../dataset/town7_dataset
```

### Additional Arguments:
- `--threshold_method` (`fixed` or `quantile`): Method to determine thresholds. Default is `quantile`.
- `--stop_threshold` (float): Fixed threshold to determine STOP label if using `fixed` method. Default is `0.1`.
- `--go_threshold` (float): Fixed threshold to determine GO label if using `fixed` method. Default is `0.2`.
- `--stop_quantile` (float): Quantile to determine STOP threshold if using `quantile` method. Default is `0.9`.
- `--go_quantile` (float): Quantile to determine GO threshold if using `quantile` method. Default is `0.1`.
- `--balance_method` (`oversample` or `undersample`): Method to balance STOP and GO labels. Default is `oversample`. Note: If either class has zero samples, oversampling or undersampling might fail. Ensure that the dataset has sufficient diversity before balancing.
- `--plot_output_path` (str): Path to save plots. Default is `../plots/dataset_images`.

### Example Command with Quantile Thresholds:
```bash
python label_dataset.py --data_path ../dataset/town7_dataset --threshold_method quantile --stop_quantile 0.9 --go_quantile 0.1
```

### Example Command with Fixed Thresholds:
```bash
python label_dataset.py --data_path ../dataset/town7_dataset --threshold_method fixed --stop_threshold 0.15 --go_threshold 0.25
```

### Note on Error Handling:
If you encounter an error like `ValueError: a must be greater than 0 unless no samples are taken`, it indicates that one of the classes has zero samples. To resolve this, consider checking the dataset distribution before attempting to balance it or adjusting the thresholds to ensure both classes are represented.
