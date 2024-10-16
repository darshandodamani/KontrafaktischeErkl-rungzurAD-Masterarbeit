import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import base64
from io import BytesIO
from PIL import Image


# Function to encode images into base64 format for hover tooltips
def encode_image(image_path):
    try:
        pil_img = Image.open(image_path)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return f'<img src="data:image/png;base64,{img_str}" width="100" height="100">'
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return "Image not found"


# Function to generate latent space visualization with image hovers and PCA reduction
def visualize_with_hover_images(csv_filename, output_file="latent_space_hover.html"):
    # Load the CSV file containing latent z values and image paths
    latent_z_values = pd.read_csv(csv_filename)

    # Extract latent vectors and perform PCA to reduce dimensionality to 2D
    z_columns = [col for col in latent_z_values.columns if col.startswith("z")]
    z_values = latent_z_values[z_columns]
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_values)

    # Add the PCA results (z1 and z2) to the DataFrame
    latent_z_values["z1"] = z_pca[:, 0]
    latent_z_values["z2"] = z_pca[:, 1]

    # Apply encoding to image paths for hover data
    latent_z_values["image_hover"] = latent_z_values["Image Path"].apply(encode_image)

    # Plot using Plotly Express, hovering shows the image, colored by label (if available)
    fig = px.scatter(
        latent_z_values,
        x="z1",
        y="z2",
        hover_data=["image_hover"],
        title="Latent Space Visualization with Hover Images",
        labels={"z1": "Latent Dimension 1", "z2": "Latent Dimension 2"},
        opacity=0.7,
    )

    # Customize the hover template to show the image
    fig.update_traces(
        marker=dict(size=8), hovertemplate="<b>Image:</b><br>%{customdata[0]}"
    )

    # Save the plot as an HTML file for interaction
    fig.write_html(output_file)
    print(f"Interactive latent space visualization saved to {output_file}")

    # Optionally, display the plot in the browser
    fig.show()


if __name__ == "__main__":
    # Specify the path to your latent z-values CSV file
    csv_filename = (
        "latent_z_values_epoch_499.csv"  # Replace with the actual CSV filename
    )
    visualize_with_hover_images(csv_filename, output_file="latent_space_hover.html")
