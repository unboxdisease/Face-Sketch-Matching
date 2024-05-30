import os
import shutil

def create_dataset(photo_folder, sketch_folder, output_folder):
    # Create output folders for each pair
    pairs = set()
    for photo_file in os.listdir(photo_folder):
        if photo_file.endswith(('.jpg', '.jpeg', '.png','.bmp')):
            print(os.path.splitext(photo_file)[0])
            pairs.add(os.path.splitext(photo_file)[0] + '_Forensic')

    for pair in pairs:
        pair_folder = os.path.join(output_folder, pair)
        os.makedirs(pair_folder, exist_ok=True)

    # Move photo and sketch files to their respective pair folders
    for photo_file in os.listdir(photo_folder):
        if photo_file.endswith(('.jpg', '.jpeg', '.png','.bmp')):
            pair_name = os.path.splitext(photo_file)[0]
            photo_src = os.path.join(photo_folder, photo_file)
            sketch_src = os.path.join(sketch_folder, photo_file)  # Adjust the extension as necessary
            print(photo_src,sketch_src)
            if os.path.exists(sketch_src):
                pair_folder = os.path.join(output_folder, pair_name + '_Forensic')
                shutil.copy2(photo_src, pair_folder+'/photo.jpg')
                shutil.copy2(sketch_src, pair_folder+'/sketch.jpg')

# Example usage
if __name__ == "__main__":
    photo_folder = "/localscratch/FaceDatabases/ForensicSketches/FullDataset/photo/"
    sketch_folder = "/localscratch/FaceDatabases/ForensicSketches/FullDataset/sketch/"
    output_folder = "/research/prip-jainkush/AdaFace/data_root/data_val"
    create_dataset(photo_folder, sketch_folder, output_folder)