import os
from PIL import Image

def resize_images(input_dir, output_dir, width=1920, height=1080):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Create the full input path
        input_path = os.path.join(input_dir, filename)
        
        # Check if the file is an image
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open an image file
            with Image.open(input_path) as img:
                # Resize the image
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                # Create the full output path
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                # Save the image in JPEG format
                img.save(output_path, 'JPEG')
                print(f'Resized and saved: {output_path}')

# Specify the input and output directories
input_directory = 'Grafika/Video_export/frames_estimated'
output_directory = 'Grafika/Video_export/frames_estimated_resized'

# Call the function
resize_images(input_directory, output_directory)
