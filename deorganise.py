import os
import shutil
import sys

# Define the input and output directories
input_dir = '/scratch0/GenFaceSketch_aligned/imgs'
output_dir = '/scratch0/Sketch_Test_Sets'

# Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Create subfolders for images and sketches
# software_dir = os.path.join(output_dir, 'Gallery_Viewed')
# sketches_dir = os.path.join(output_dir, 'Viewed_probes_tufts')
# if not os.path.exists(software_dir):
#     os.makedirs(software_dir)
# if not os.path.exists(sketches_dir):
#     os.makedirs(sketches_dir)

# Iterate over the subfolders in the input directory
i=0
for id_dir in os.listdir(input_dir):
        id_dir_path = os.path.join(input_dir, id_dir)
        for file in os.listdir(id_dir_path):
            if not('_' in file):
                # img_name = file[:-4]
                # paths.append(os.path.join(id_dir_path, file)) 

                shutil.copy(os.path.join(id_dir_path, file), os.path.join('/scratch0/FS-SGC/imgs/' + file[:-4], file))
        # elif file.split('_')[1] == 'CS':
        #     if not os.path.exists(os.path.join(sketches_dir, f'{id_dir}')):
        #         os.makedirs(os.path.join(sketches_dir, f'{id_dir}'), exist_ok=True)
        #     shutil.copy(os.path.join(id_dir_path, file), os.path.join(sketches_dir, f'{id_dir}/{id_dir}.jpg'))
