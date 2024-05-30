import os
import shutil
import sys

def reorganize_dataset(input_dir, output_dir):
    """
    Reorganize the directory structure of a dataset such that each image has its own subfolder named as the image name.

    Args:
    - input_dir (str): Path to the input directory containing the dataset.
    - output_dir (str): Path to the output directory where the reorganized dataset will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    # Iterate through each file in the input directory
    for root, dirs, files in os.walk(input_dir):
        # print(files)
        for filename in files:
            # Check if the file is an image file
            if filename.split('_')[1] == 'RGB':
                # Construct the path to the input file
                input_file_path = os.path.join(root, filename)
                # print(input_file_path)
                
                # Create a subfolder with the same name as the image file (without extension)
                # image_name = os.path.splitext(filename)[0] + '_tufts'
                image_name = input_file_path.split('/')[-2]
                
                image_subfolder = os.path.join(output_dir, image_name)
                os.makedirs(image_subfolder, exist_ok=True)
                
                # Construct the path to the output file
                output_file_path = os.path.join(image_subfolder, filename)
                
                # Move the file to the subfolder
                shutil.copy(input_file_path, output_file_path)
                i+=1
                print(f"Moved '{filename}' to '{image_subfolder}'.")
                if i >= 10000:
                    sys.exit()


# Example usage:
input_directory = "//localscratch/FaceDatabases/TD_CS_aligned/"
output_directory = "/localscratch/FaceDatabases_aligned/Gallery_Viewed/"
reorganize_dataset(input_directory, output_directory)
