from PIL import Image
import os

# Set the paths to your folders
folder1 = 'YOLOX_outputs/inference_s'
folder2 = 'YOLOX_outputs/inference_l'
output_folder = 'YOLOX_outputs/sidebyside'

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
        new_width = image1.width + image2.width
        new_height = max(image1.height, image2.height)
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste the images side by side
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

        # Save the new image
        output_filename = os.path.join(output_folder, filename)
        new_image.save(output_filename)