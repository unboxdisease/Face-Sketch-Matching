import net
import torch
import os
from face_alignment import align
import numpy as np
import argparse  
from tqdm import tqdm
from data import DatasetInference
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch.utils import data
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection



image_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32').to(
            'cuda:0', dtype=torch.float16
        )
clip_image_processor = CLIPImageProcessor()


def genuine_pairs(embeddings_dict, subjs, symmetric=False):
    print('calculating genuine scores...')
    genuine_scores = []
    for subj in subjs:
        n_embeddings = len(embeddings_dict[subj])
        for e_idx1 in range(n_embeddings):
            path1, embedding1 = embeddings_dict[subj][e_idx1]
            if symmetric:
                for e_idx2 in range(e_idx1 + 1, n_embeddings):
                    path2, embedding2 = embeddings_dict[subj][e_idx2]
                    gen_score = np.dot(embedding1, embedding2)
                    genuine_scores.append((path1, path2, gen_score))
            else:
                for e_idx2 in range(n_embeddings):
                    path2, embedding2 = embeddings_dict[subj][e_idx2]
                    if path2 != path1:
                        gen_score = np.dot(embedding1, embedding2)
                        genuine_scores.append((path1, path2, gen_score))
    print('len genuine scores: ', len(genuine_scores))
    return np.array(genuine_scores)

def imposter_pairs(embeddings_dict, subjs, all_imposter=False, symmetric=False):
    imposter_scores = []
    n_subjs = len(subjs)

    if not all_imposter:
        if symmetric:
            print('calculating symmetric first imposters only')
            for subj_idx1 in range(n_subjs):
                subj1 = subjs[subj_idx1]
                for subj_idx2 in range(subj_idx1+1, n_subjs):
                    subj2 = subjs[subj_idx2]
                    path1, embedding1 = embeddings_dict[subj1][0]
                    path2, embedding2 = embeddings_dict[subj2][0]
                    imp_score = np.dot(embedding1, embedding2)
                    imposter_scores.append((path1, path2, imp_score))
        else:
            print('calculating first imposter scores')
            for subj_idx1 in range(n_subjs):
                subj1 = subjs[subj_idx1]
                for subj_idx2 in range(n_subjs):
                    subj2 = subjs[subj_idx2]
                    if subj1 != subj2:
                        path1, embedding1 = embeddings_dict[subj1][0]
                        path2, embedding2 = embeddings_dict[subj2][0]
                        imp_score = np.dot(embedding1, embedding2)
                        imposter_scores.append((path1, path2, imp_score))
        print('len imposter scores: ', len(imposter_scores))
        return np.array(imposter_scores)
    if not symmetric:
        print('calculating all imposter scores')
        for subj_idx1 in range(n_subjs):
            subj1 = subjs[subj_idx1]
            for subj_idx2 in range(n_subjs):
                subj2 = subjs[subj_idx2]
                if subj1 != subj2:
                    for path1, embedding1 in embeddings_dict[subj1]:
                        for path2, embedding2 in embeddings_dict[subj2]:
                            imp_score = np.dot(embedding1, embedding2)
                            imposter_scores.append((path1, path2, imp_score))
    else:
        print('calculating all symmetric imposter scores')
        for subj_idx1 in range(n_subjs):
            subj1 = subjs[subj_idx1]
            for subj_idx2 in range(subj_idx1 + 1, n_subjs):
                subj2 = subjs[subj_idx2]
                for path1, embedding1 in embeddings_dict[subj1]:
                    for path2, embedding2 in embeddings_dict[subj2]:
                        imp_score = np.dot(embedding1, embedding2)
                        imposter_scores.append((path1, path2, imp_score))
    print('len imposter scores: ', len(imposter_scores))
    return np.array(imposter_scores)

def roc_threshold(imposter_scores, genuine_scores, save_dir, compute_EER=False):
    if type(genuine_scores) != list:
        imposter_scores = imposter_scores.tolist()
        genuine_scores = genuine_scores.tolist()
    
    imposter_labels = [0 for ls in imposter_scores]
    genuine_labels = [1 for ss in genuine_scores]
    y = genuine_labels + imposter_labels
    y = np.array(y)
    scores = genuine_scores + imposter_scores
    scores = np.array(scores)
    fpr, tpr, thresholds = roc_curve(y, scores)

    fnr = 1 - tpr
    if compute_EER:
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # print("EER = {} at threshold = {}".format(EER, eer_threshold))

    found = False
    optimal_threshold_idx = []
    for j, fpr_j in enumerate(fpr):
        if fpr_j > 0.0001 and not found:
            print("TAR = {}% @ FAR: {}% Threshold = {}".format(tpr[j - 1] * 100, fpr[j - 1] * 100, thresholds[j - 1]))
            optimal_threshold_idx.append(j-1)
            found = True
        if found and fpr_j > .001:
            print("TAR = {}% @ FAR: {}% Threshold = {}".format(tpr[j - 1] * 100, fpr[j - 1] * 100, thresholds[j - 1]))
            optimal_threshold_idx.append(j-1)
            return fpr, tpr, thresholds, optimal_threshold_idx


def matching_evaluation(embeddings_dict, all_imposter=True, symmetric=False): 
    subjs = list(embeddings_dict.keys())
    gen = genuine_pairs(embeddings_dict, subjs, symmetric)
    imp = imposter_pairs(embeddings_dict, subjs, all_imposter, symmetric)
    return gen, imp

def save_roc_and_hist(imposter_scores, genuine_scores, fpr, tpr, roc_path, hist_path):
    plt.figure()
    plt.plot(fpr*100, tpr*100, linewidth=2.0)
    plt.xlim([0, 0.1])
    plt.ylim([0, 100])
    plt.title("ROC")
    plt.ylabel("TAR or 1-FRR (%)")
    plt.xlabel("FAR (%)")
    plt.grid()
    print('saving ROC')
    plt.savefig(roc_path)

    plt.figure()
    plt.hist(imposter_scores, bins='auto', density=True, alpha=0.8)
    plt.hist(genuine_scores, bins='auto', density=True, alpha=0.8)
    plt.title("Score Histogram")
    plt.legend(['Imposters', 'Genuines'])
    print("saving histogram")
    plt.savefig(hist_path)

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
    'ir_101':"pretrained/adaface_ir101_webface4m.ckpt",
    'ir_101-sketch':"experiments/ir101_overfit_04-09_0/last.ckpt",
    'ir_50-dcface':"experiments/ir50_dcface_adaface_03-15_0/epoch=14-step=16125.ckpt",
    'ir_50-dcface_watermarked':"experiments/ir50_dcface_watermarked_adaface_03-15_0/epoch=15-step=17200.ckpt",
    'ir_50-webface':"experiments/ir50_webface_adaface_03-14_1/epoch=21-step=21098.ckpt",
    'ir_50-webface_watermarked':"experiments/ir50_webface_watermarked_adaface_03-14_0/epoch=23-step=23016.ckpt",
    'ir_50-ms1m':"experiments/ir50_ms1v2_03-17_0/epoch=24-step=284325.ckpt",
    'ir_50-ms1m_watermarked':"experiments/ir50_ms1v2_watermark_03-17_0/epoch=23-step=272952.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    model_name = architecture
    architecture = architecture.split('-')[0]
    print(architecture)
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[model_name])['state_dict']
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

def main(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = load_pretrained_model(args.ckpt).to(device)

    # symmetric = not args.nonsymmetric
    # all_imposter = not args.first_imposter_only
    save_dir = args.save_dir

    # paths = get_image_paths(args.d)
    # valid_dataset = DatasetInference(paths, resolution=112, in_channels=3)
    # validloader = data.DataLoader(valid_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=False,
    #                               num_workers=2,
    #                               drop_last=False)
    # embeddings_dict = defaultdict(list)
    # print('extracting embeddings...')
    # with torch.no_grad():
    #     for datas, paths in tqdm(validloader):
    #         datas = datas.to(device)
    #         print(paths)
    #         features, _ = model(datas)

    #         for feature, path in zip(features, paths):
    #             #normalize CNN embedding
    #             t_i = path.split("/")[-2]
    #             e_i = feature.detach().cpu().numpy()
    #             # e_i_norm = np.sqrt(e_i.dot(e_i))
    #             # e_i = (1.0 / e_i_norm) * e_i
    #             embeddings_dict[t_i].append((path, e_i))

    #             # outpath = path.replace(args.d, args.outd).split('.')[0] + '.npy'
    #             # outdir = '/'.join(outpath.split("/")[:-1])
    #             # if not os.path.isdir(outdir):
    #             #     os.makedirs(outdir)
    #             # np.save(outpath, e_i)

    print('computing scores...')
    # genuine_scores, imposter_scores = matching_evaluation(embeddings_dict, all_imposter, symmetric)
    genuine_scores = np.load('./Assets2/ROC_Synthetic/genuine_scores_.npy')
    imposter_scores = np.load('./Assets2/ROC_Synthetic/imposter_scores_.npy')
    print(genuine_scores.size,imposter_scores.size)
    fpr, tpr, thresholds, optimal_threshold = roc_threshold(imposter_scores[:,2].astype(np.float32), genuine_scores[:,2].astype(np.float32), save_dir)
    
    if save_dir is not None:
        print('saving results...')
        gen_path = os.path.join(save_dir, 'gen_scores.npy')
        imp_path = os.path.join(save_dir, 'imp_scores.npy')
        roc_path = os.path.join(save_dir, 'roc.png')
        hist_path = os.path.join(save_dir, 'hist.png')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # np.save(gen_path, genuine_scores)
        # np.save(imp_path, imposter_scores)
        print(genuine_scores[:,2])
        save_roc_and_hist(imposter_scores[:,2], genuine_scores[:,2], fpr, tpr, roc_path, hist_path)
    else:
        print('not saving results')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--ckpt", default="ir_101", help="model ckpt", type=str)
    parser.add_argument("--d", default="/research/prip-jainkush/AdaFace/data_root/data_test/Non-Viewed/", help="Input dir", type=str)
    # parser.add_argument("--d", default="/localscratch/groszste/FaceDatabases/CASIA-Webface_prip_aligned_mtcnn/val_2imgsperID_watermarked", help="Input dir", type=str)
    parser.add_argument('--first_imposter_only', action='store_true', help='Compute first imposters')
    parser.add_argument('--nonsymmetric', action='store_true', help='Compute entire similarity matrix, rather than just upper triangle')
    parser.add_argument('--save_dir', default='./Assets2/ROC_Synthetic', help='save dir')
    parser.add_argument("--batch_size", default=512, help="batch size", type=int)
    args = parser.parse_args()
    main(args)
