import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the latent space data
latent_df = pd.read_csv("latent_z_values_epoch_440.csv")

# Dynamically add the `label` column based on the `Image Path`
def infer_label(image_path):
    if "STOP" in image_path:
        return "STOP"
    elif "GO" in image_path:
        return "GO"
    elif "RIGHT" in image_path:
        return "RIGHT"
    elif "LEFT" in image_path:
        return "LEFT"
    else:
        return "UNKNOWN"

latent_df["label"] = latent_df["Image Path"].apply(infer_label)

# Extract latent vectors and labels
latent_vectors = latent_df.iloc[:, 1:-1].values  # All columns except 'Image Path' and 'label'
labels = latent_df["label"].values

# Perform t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

# Define a colormap for distinct colors
unique_labels = np.unique(labels)
colors = cm.tab20(np.linspace(0, 1, len(unique_labels)))

# Create a scatter plot with clear colors
plt.figure(figsize=(12, 8))
for i, label in enumerate(unique_labels):
    mask = labels == label
    plt.scatter(
        latent_2d[mask, 0],
        latent_2d[mask, 1],
        label=label,
        c=[colors[i]],
        alpha=0.8,
        edgecolor='k',
        s=50,
    )

plt.title("Latent Space Visualization (t-SNE) for 4 Classes", fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend(title="Class Labels", fontsize=10, loc="best")
plt.grid(True, alpha=0.3)

# Save and display the visualization
plt.savefig("results/latent_space_tsne_visualization_distinct_colors.png", dpi=300)
print("Latent space visualization saved at results/latent_space_tsne_visualization_distinct_colors.png")
plt.show()
