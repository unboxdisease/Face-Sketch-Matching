import os
import shutil

# Define the input and output directories
input_dir = '/localscratch/FaceDatabases/IIITD_SketchDatabase_aligned/Viewed_sketch/IIIT-D'
output_dir = '/localscratch/FaceDatabases/Test_Set_Combined'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all the image and sketch files
image_files = [f for f in os.listdir(os.path.join(input_dir, 'photo')) if f.endswith('.jpg')]
sketch_files = [f for f in os.listdir(os.path.join(input_dir, 'sketch')) if f.endswith('.jpg')]

# Create a dictionary to map IDs to file names
id_to_file = {}
for file in image_files:
    id = os.path.splitext(file)[0]
    id_to_file[id] = {'image': file}
for file in sketch_files:
    id = os.path.splitext(file)[0].replace('s','p')
    if id in id_to_file:
        id_to_file[id]['sketch'] = file
    else:
        id_to_file[id] = {'sketch': file}

# Create subfolders for each ID and move the corresponding files into them
for id, files in id_to_file.items():
    id_dir = os.path.join(output_dir, id + '_IIITD')
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)
    # if 'image' in files:
        # shutil.copy(os.path.join(input_dir, 'photo', files['image']), os.path.join(id_dir, f'{id}_photo.jpg'))
    if 'sketch' in files:
        shutil.copy(os.path.join(input_dir, 'sketch', files['sketch']), os.path.join(id_dir, f'{id}_sketch_viewed.jpg'))
