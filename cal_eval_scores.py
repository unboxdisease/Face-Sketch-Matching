import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from data import DatasetInference
import net
import numpy as np

adaface_models = {
    'ir_101':"pretrained/adaface_ir101_webface4m.ckpt",
    'ir_101-base':"pretrained/adaface_ir101_webface4m.ckpt",
    'ir_101-real':"experiments/ir101_custom_adaface_04-07_3/last.ckpt",
    'ir_101-synthetic':"experiments/ir101_synthetic_sketches_05-01_0/last.ckpt",
    'ir_101-synthreal':"experiments/synth+real/last-v2.ckpt"
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    # model_name = architecture
    # architecture = architecture.split('-')[0]
    # print(architecture)
    # assert architecture in adaface_models.keys()
    model = net.build_model('ir_101')
    statedict = torch.load(architecture)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

def get_image_paths(root_folder, image_extensions=['png','jpg','jpeg','bmp']):
    paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                path = os.path.join(root, file)
                paths.append(path)
                print(path)
    return paths
device = 'cuda'

# Load the models
# models = {
#     'ir_101-base': load_pretrained_model('ir_101-base').to(device),
#     'ir_101-real': load_pretrained_model('ir_101-real').to(device),
#     'ir_101-synthetic': load_pretrained_model('ir_101-synthetic').to(device),
#     'ir_101-synthreal': load_pretrained_model('ir_101-synthreal').to(device)
# }

# Load the probe and gallery images
probe_paths = get_image_paths('/localscratch/FaceDatabases_aligned/Combined_probes')
gallery_paths = get_image_paths('/localscratch/FaceDatabases_aligned/Gallery_Viewed')

# Create the data loaders
probe_dataset = DatasetInference(probe_paths, resolution=112, in_channels=3)
gallery_dataset = DatasetInference(gallery_paths, resolution=112, in_channels=3)

probe_loader = DataLoader(probe_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=False)



model_pathsfinal = [
"experiments/ir101_ms1m_baseline_04-17_4/last.ckpt",
'experiments/ir101_GenFaceSketch75++_05-25_0/last.ckpt',
'experiments/ir101_FS-SGC_05-25_0/epoch=32-step=79868.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=40-step=77113.ckpt'
]



# Calculate the similarity scores
for model_name in model_pathsfinal:
    model = load_pretrained_model(model_name).to(device)
    folder = model_name.split('/')[-2]
    output_file = f'./Assets2/{folder}/Viewed_scores_final.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Calculate gallery embeddings only once
    gallery_embeddings = []
    gallery_paths = []
    for gallery_datas, gallery_paths_batch in gallery_loader:
        gallery_datas = gallery_datas.to(device)
        gallery_embedding, _ = model(gallery_datas)
        gallery_embeddings.extend(gallery_embedding.detach().cpu().numpy())
        gallery_paths.extend(gallery_paths_batch)

    with open(output_file, 'a+') as f:
        for probe_datas, probe_paths in tqdm(probe_loader):
            probe_datas = probe_datas.to(device)
            probe_embedding, _ = model(probe_datas)
            probe_embedding = probe_embedding.detach().cpu().numpy()
            for probe_path in probe_paths:
                for gallery_path, gallery_embedding in zip(gallery_paths, gallery_embeddings):
                    similarity_score = np.dot(probe_embedding, gallery_embedding)

                    f.write(f"{probe_path},{gallery_path},{similarity_score}\n")
