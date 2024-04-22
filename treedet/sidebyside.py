from PIL import Image, ImageDraw

import os

# Set the paths to your folders
folder1 = "YOLOX_outputs/inf_sussy"
folder2 = "YOLOX_outputs/inf_sussy_frozen"
output_folder = "YOLOX_outputs/ds_sidebyside_sussy"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of image filenames in the first folder
filenames = os.listdir(folder1)

for filename in filenames:
    # Construct the full file path for both images
    file1 = os.path.join(folder1, filename)
    file2 = os.path.join(folder2, filename)

    if os.path.isfile(file1) and os.path.isfile(file2):
        # Open the images
        image1 = Image.open(file1)
        image2 = Image.open(file2)

        # Create a new image with appropriate dimensions
        new_width = max(image1.width, image2.width)
        new_height = image1.height + image2.height

        new_image = Image.new("RGB", (new_width, new_height))

        # Paste the images above eachother
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))

        # add text labels:
        text_height = 20
        draw = ImageDraw.Draw(new_image)

        draw.rectangle([(0, 0), (75, 20)], fill="#fff")
        draw.text((10, 0), "cana100", fill="black")

        draw.rectangle([(0, image1.height), (75, image1.height + 14)], fill="#fff")
        draw.text((10, image1.height), "wiki100", fill="black")

        # Save the new image
        output_filename = os.path.join(output_folder, filename)
        new_image.save(output_filename)
