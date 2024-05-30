import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image
from torch.utils import data
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from face_alignment import align
import net 



adaface_models = {
    'ir_101':"pretrained/adaface_ir101_webface4m.ckpt",
}
image_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32').to(
            'cuda:0', dtype=torch.float16
        )
clip_image_processor = CLIPImageProcessor()


def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model
model = load_pretrained_model('ir_101').to('cuda:0')
def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float().to('cuda:0')
    return tensor




class Dataset(data.Dataset):
    def __init__(self, paths):
        self.imgs = paths

    def __getitem__(self, index):
        img_path = self.imgs[index]
        return img_path

    def __len__(self):
        return len(self.imgs)

def extract_embs_adaface(paths):
    
    features = []
    for path in paths:
        # aligned_rgb_img = align.get_aligned_face(path)
        # if aligned_rgb_img is None:
        #     continue
        imgs = Image.open(path).convert('RGB')
        bgr_tensor_input = to_input(imgs)
        feature, _ = model(bgr_tensor_input)
        feature = torch.squeeze(feature).cpu().detach().numpy()
        features.append(feature)
    return np.array(features)

def extract_embs_clip(paths):

    imgs = [Image.open(path).convert('RGB') for path in paths]
    clip_image = clip_image_processor(images=imgs, return_tensors="pt").pixel_values
    clip_image_embeds = image_encoder(clip_image.to('cuda:0', dtype=torch.float16)).image_embeds
    feature2 = torch.squeeze(clip_image_embeds).cpu().detach().numpy()
    
    # embeds = img2vec.get_vec(imgs)
    return feature2

def extract_embs_vgg(paths):
    img2vec = Img2Vec(cuda=True)
    imgs = [Image.open(path).convert('RGB') for path in paths]
    embeds = img2vec.get_vec(imgs)
    return embeds

def get_image_paths(root_folder, image_extensions=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff']):
    IDs = defaultdict(list)
    paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                path = os.path.join(root, file)
                ID = path.split('/')[-2]
                IDs[ID].append(path)
                print(path)
                paths.append(path)
    return IDs, paths

def create_tsne_plot(embeddings, labels, outd):
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1)  # Adjust perplexity to be within a valid range
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        c=label[-2]
        m=label[-1]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],c=c, marker=m, label=f'{label[:-3]}', s=100)
    plt.title('t-SNE plot of '+outd+' features')
    plt.legend()
    plt.savefig(os.path.join('./Assets2/tsne/', 'tsne_plot_'+outd+'_size.png'))
    plt.show()

def create_tsne_plot_3d(embeddings, labels, outd):
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1)  # Adjust perplexity to be within a valid range
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=0)
    embeddings_3d = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        c=label[-2]
        m=label[-1]
        ax.scatter(embeddings_3d[indices, 0], embeddings_3d[indices, 1], embeddings_3d[indices, 2],c=c, marker=m, label=f'{label[:-3]}')
    ax.set_title('3D t-SNE plot of '+outd+' features')
    ax.legend()
    plt.savefig(os.path.join('./Assets2/tsne/', 'tsne_plot_'+outd+'_3d.png'))
    plt.show()

def main(args):
    d = args.d
    outd = args.outd
    emb = args.emb

    ID_dict, paths = get_image_paths(d)
    dataset = Dataset(paths)
    loader = data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)

    embeddings = []
    labels = []
    for paths_batch in tqdm(loader):
        if emb == 'adaface':    
            embeds_batch = extract_embs_adaface(paths_batch)
        if emb == 'vgg':    
            embeds_batch = extract_embs_vgg(paths_batch)
        if emb == 'clip':    
            embeds_batch = extract_embs_clip(paths_batch)
        # print(embeds_batch.shape)
        embeddings.extend(embeds_batch)
        for path in paths_batch:
            label = path.split('/')[-2]  # Extract label from the directory structure
            labels.append(label)

    embeddings = np.array(embeddings)
    print(embeddings.shape)
    labels = np.array(labels)

    create_tsne_plot(embeddings, labels, emb )
    create_tsne_plot_3d(embeddings, labels, emb )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default='/research/prip-jainkush/Datasets/CUHK/' , type=str, help="Input directory containing images")
    parser.add_argument("--outd",  default='/research/prip-jainkush/Adaface/Assets2/tsne/' , type=str, help="Output directory to save t-SNE plot")
    parser.add_argument("--emb",  default='vgg' , type=str, help="Embedding type")
    args = parser.parse_args()
    main(args)
