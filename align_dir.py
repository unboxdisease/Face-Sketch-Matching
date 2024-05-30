import net
import torch
import os
from face_alignment import align
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

def get_image_paths(root_folder, image_extensions=['bmp', 'png','jpg','jpeg', 'tif', 'tiff']):
    paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                path = os.path.join(root, file)
                paths.append(path)
    return paths

def center_crop(img, out_w, out_h):
    h,w = img.shape[:2]
    x = w // 2
    y = h // 2
    x1 = x - (out_w // 2)
    x2 = x + (out_w // 2)
    y1 = y - (out_h // 2)
    y2 = y + (out_h // 2)

    crop = img[y1:y2, x1:x2]
    return crop

def main(args):
    d = args.d
    outd = args.outd
    paths = get_image_paths(d)
    paths.sort()
    failed = 0
    for path in tqdm(paths):
        outpath = path.replace(d, outd)
        if not os.path.exists(outpath):
            outdir = '/'.join(outpath.split("/")[:-1])
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            aligned_rgb_img = align.get_aligned_face(path)
            if aligned_rgb_img is not None:
                aligned_rgb_img.save(outpath)
            else:
                failed += 1
    print("num failed: ", failed)
    paths_ = get_image_paths(outd)
    assert len(paths) == len(paths_)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, default="/research/prip-jainkush/AdaFace/data_root/real_test", help='set input image directory')
    parser.add_argument('--outd', type=str, default="/research/prip-jainkush/AdaFace/data_root/real_test_aligned", help='set output image directory')
    args = parser.parse_args()
    main(args)