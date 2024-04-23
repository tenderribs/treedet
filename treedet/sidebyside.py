import os
import math

from PIL import Image, ImageDraw, ImageFont

folders = [
    "inf_yolox_s_canawiki325_sparse_nol1.onnx",
    "inf_yolox_l_canawikisparse325_late_l1.onnx",
    "inf_yolox_l_canawikisparse325_l1.onnx",
]

labels = [
    "S sparse no L1",
    "L sparse late L1",
    "L sparse full L1",
]

assert len(folders) == len(labels)

folder_base = "YOLOX_outputs"
output_folder = "YOLOX_outputs/sidebyside_mark_forest"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of image filenames in the first folder
filenames = os.listdir(os.path.join(folder_base, folders[0]))
filenames = [
    file for file in filenames if file.endswith(".jpg") or file.endswith(".png")
]

cols = 2

for file in filenames:
    print(f"processing {file}")

    files = [os.path.join(folder_base, folder, file) for folder in folders]
    images = [Image.open(file) for file in files]

    # Create a new image with appropriate dimensions
    width = images[0].width
    height = images[0].height

    collage_width = width * cols
    collage_height = height * (math.ceil(len(images) / float(cols)))
    new_image = Image.new("RGB", (collage_width, collage_height))

    # for text labels:
    draw = ImageDraw.Draw(new_image)
    font_size = 20  # Adjust as needed
    font = ImageFont.truetype("Arial.ttf", font_size)

    for idx, (image, label) in enumerate(zip(images, labels)):
        # Paste the images above eachother
        corner_x = width * (idx % cols)
        corner_y = height * (idx // cols)
        new_image.paste(image, (corner_x, corner_y))

        draw.rectangle(
            [(corner_x, corner_y), (corner_x + 250, corner_y + 30)], fill="#fff"
        )
        draw.text((corner_x + 10, corner_y), label, fill="black", font=font)

    output_filename = os.path.join(output_folder, file)
    new_image.save(output_filename)
