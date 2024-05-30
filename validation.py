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
from backbones import get_model
import cv2



def genuine_pairs(embeddings_dict, subjs, symmetric=False):
    print('calculating genuine scores...')
    genuine_scores = []
    for subj in subjs:
        n_embeddings = len(embeddings_dict[subj])
        for e_idx1 in range(n_embeddings):
            path1, embedding1 = embeddings_dict[subj][e_idx1]
            # if not('sketch' in path1):
            #     continue
            if symmetric:
                for e_idx2 in range(e_idx1 + 1, n_embeddings):
                    path2, embedding2 = embeddings_dict[subj][e_idx2]
                    gen_score = np.dot(embedding1, embedding2)
                    genuine_scores.append((path1, path2, gen_score))
            else:
                for e_idx2 in range(n_embeddings):
                    path2, embedding2 = embeddings_dict[subj][e_idx2]
                    # if not('viewed' in path2):
                    #     continue
                    if path2 != path1:
                        gen_score = np.dot(embedding1, embedding2)
                        genuine_scores.append((path1, path2, gen_score))
    print('len genuine scores: ', len(genuine_scores))
    return np.array(genuine_scores)

def imposter_pairs(embeddings_dict, subjs, all_imposter=True, symmetric=False):
    imposter_scores = []
    n_subjs = len(subjs)
    all_imposter = True
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
                        if not('photo' in path1):
                            continue
                        for path2, embedding2 in embeddings_dict[subj2]:
                            if 'photo' in path2:
                                continue
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

def roc_threshold(imposter_scores, genuine_scores, save_dir, epoch,compute_EER=False):
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
            epoch_score.append((tpr[j - 1] * 100, epoch))
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
    'ir_101':"experiments/ir101_ms1m_baseline_04-17_4/last.ckpt",
    'ir_101-sketch':"experiments/ir101_custom_adaface_04-07_3/last.ckpt",
    'ir_101-synthsketch':"experiments/ir101_synthetic_sketches_05-08_0/last.ckpt",
    'ir_101-synthreal':"experiments/synth+real/last-v2.ckpt",
    'ir_50-dcface':"experiments/ir50_dcface_adaface_03-15_0/epoch=14-step=16125.ckpt",
    'ir_50-dcface_watermarked':"experiments/ir50_dcface_watermarked_adaface_03-15_0/epoch=15-step=17200.ckpt",
    'ir_50-webface':"experiments/ir50_webface_adaface_03-14_1/epoch=21-step=21098.ckpt",
    'ir_50-webface_watermarked':"experiments/ir50_webface_watermarked_adaface_03-14_0/epoch=23-step=23016.ckpt",
    'ir_50-ms1m':"experiments/ir50_ms1v2_03-17_0/epoch=24-step=284325.ckpt",
    'ir_50-ms1m_watermarked':"experiments/ir50_ms1v2_watermark_03-17_0/epoch=23-step=272952.ckpt",
}

def save_failure_cases(genuine_scores, threshold, save_dir):
    failure_cases = [score for score in genuine_scores if float(score[2]) < threshold]
    normal_cases = [score for score in genuine_scores if float(score[2]) > threshold]
    failure_cases_dir = os.path.join(save_dir, 'failure_cases')
    normal_cases_dir = os.path.join(save_dir, 'normal_cases')
    if not os.path.exists(failure_cases_dir):
        os.makedirs(failure_cases_dir)
    if not os.path.exists(normal_cases_dir):
        os.makedirs(normal_cases_dir)
    
    num = 0
    for pair in normal_cases:
        path1, path2, score = pair
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img1 = cv2.resize(img1,(112,112))
        img2 = cv2.resize(img2,(112,112))
        concat_img = np.concatenate((img1, img2), axis=1)
        num+=1
        cv2.imwrite(os.path.join(normal_cases_dir, f'{path1.split("/")[-1]}_{path2.split("/")[-1]}.jpg'), concat_img)
        if num>=100:
            break
    num = 0
    for pair in failure_cases:
        path1, path2, score = pair
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img1 = cv2.resize(img1,(112,112))
        img2 = cv2.resize(img2,(112,112))
        concat_img = np.concatenate((img1, img2), axis=1)
        num+=1
        cv2.imwrite(os.path.join(failure_cases_dir, f'{path1.split("/")[-1]}_{path2.split("/")[-1]}.jpg'), concat_img)
        if num>=100:
            break

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict

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
                # print(path)
    return paths


    # img = np.transpose(img, (2, 0, 1))
    # img = torch.from_numpy(img).unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    # net = get_model(name, fp16=False)
    # net.load_state_dict(torch.load(weight))
    # net.eval()

model_paths25 = [

'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=30.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=43-step=90604.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=28.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=34.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=27-step=75772.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=38-step=85969.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=41-step=88750.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=31.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=29.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=35-step=83188.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=51-step=98020.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=44-step=91531.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=46-step=93385.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=60-step=106363.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=42-step=89677.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=36-step=84115.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=47.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=26-step=74845.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=33.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=58-step=104509.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=50-step=97093.ckpt',

'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=48-step=95239.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=45-step=92458.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=40.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=59-step=105436.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=32.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=39-step=86896.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=57-step=103582.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=37-step=85042.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=56-step=102655.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=49-step=96166.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=54.ckpt'


]
model_paths50 = [
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=29-step=81346.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=30-step=83203.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=31-step=85060.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=32-step=86917.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=33-step=88774.ckpt'
]
model_paths75 = [
'experiments/ir101_GenFaceSketch75_4_05-19_0/epoch=31.ckpt',
'experiments/ir101_GenFaceSketch75_4_05-19_0/epoch=32.ckpt',
'experiments/ir101_GenFaceSketch75_4_05-19_0/epoch=33-step=96182.ckpt',
'experiments/ir101_GenFaceSketch75_4_05-19_0/epoch=34-step=98965.ckpt',
]
model_paths100 = [
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=28-step=85051.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=29-step=88762.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=31-step=96184.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=32-step=99895.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=33-step=103606.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=34-step=107317.ckpt'


]
model_paths_real = [


'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=51-step=87412.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=53-step=88450.ckpt',


'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=56-step=90007.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=57-step=90526.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=58-step=91045.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=59-step=91564.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=60-step=92083.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=61-step=92602.ckpt'
]

model_pathsinfo = [

'experiments/ir101_InformativeImgs_05-12_0/epoch=27-step=74344.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=52-step=79669.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=36-step=76261.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=29-step=74770.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=48-step=78817.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=40-step=77113.ckpt'


]

model_pathsfinal = [

'experiments/ir101_GenFaceSketch25_05-11_0/epoch=31-step=79486.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=29-step=81346.ckpt',
'experiments/ir101_GenFaceSketch75_05-11_0/epoch=35-step=101768.ckpt',
'experiments/ir101_synthetic_exp1_all_mug_05-09_0/epoch=46-step=93847.ckpt',
'experiments/ir101_GenFaceSketchReal++_05-14_1/epoch=60-step=92083.ckpt',
'experiments/ir101_InformativeImgs_05-12_0/epoch=40-step=77113.ckpt'
]

model_pathsfinal2 = [
'pretrained/adaface_ir101_webface4m.ckpt',
"experiments/ir101_ms1m_baseline_04-17_4/last.ckpt",
'experiments/ir101_GenFaceSketch25_05-11_0/epoch=29-step=77630.ckpt',
'experiments/ir101_GenFaceSketch25_2_05-19_0/epoch=29-step=77634.ckpt',
'experiments/ir101_GenFaceSketch25_3_05-19_0/epoch=29.ckpt',
'experiments/ir101_GenFaceSketch25_4_05-19_0/epoch=32.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=29-step=81346.ckpt',
'experiments/ir101_GenFaceSketch50_2_05-19_0/epoch=29.ckpt',
'experiments/ir101_GenFaceSketch50_3_05-22_0/epoch=30-step=83193.ckpt',
'experiments/ir101_GenFaceSketch50_4_05-19_0/epoch=31.ckpt',
'experiments/ir101_GenFaceSketch75_05-11_0/epoch=31-step=90628.ckpt',
'experiments/ir101_GenFaceSketch75_3_05-22_0/epoch=33-step=96182.ckpt',
'experiments/ir101_GenFaceSketch75_2_05-19_0/epoch=31.ckpt',
'experiments/ir101_GenFaceSketch75_4_05-19_0/epoch=31.ckpt',
'experiments/ir101_GenFaceSketch100_05-22_0/epoch=30-step=92473.ckpt'
]


exp2 = [
"experiments/ir101_ms1m_baseline_04-17_4/last.ckpt",
'experiments/ir101_GenFaceSketch25_05-11_0/epoch=31-step=79486.ckpt',
'experiments/ir101_GenFaceSketch50_05-11_0/epoch=29-step=81346.ckpt',
'experiments/ir101_GenFaceSketch75_05-11_0/epoch=35-step=101768.ckpt',
'experiments/ir101_synthetic_exp1_all_mug_05-09_0/epoch=46-step=93847.ckpt',
]
exp3 = [
"experiments/ir101_ms1m_baseline_04-17_4/last.ckpt",
'experiments/ir101_GenFaceSketch25_05-11_0/epoch=32-step=80414.ckpt',
# ''experiments/ir101_GenFaceSketch50_05-11_0/epoch=37-step=96202.ckpt',
# 'experiments/ir101_GenFaceSketch75_05-11_0/epoch=34-step=98983.ckpt',
# 'experiments/ir101_synthetic_exp1_all_mug_05-09_0/epoch=46-step=93847.ckpt',
'experiments/ir101_GenFaceSketch+real_05-10_0/epoch=71-step=117848.ckpt',



]



# labels = ['ms1m face database', '25% of synthetic data', '25++', 'infoDrawing']

def main(args):
    
    # weight = './pretrained/backbone.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    symmetric = not args.nonsymmetric
    all_imposter = not args.first_imposter_only
    save_dir = args.save_dir
    fprs = []
    tprs = []
    labels = []
    paths = get_image_paths(args.d)
    valid_dataset = DatasetInference(paths, resolution=112, in_channels=3)
    validloader = data.DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=False)
    

    for model_path in model_pathsfinal2:
            # Load the model
        model = load_pretrained_model(model_path).to(device)

        # Extract embeddings and compute scores
        embeddings_dict = defaultdict(list)
        with torch.no_grad():
            for datas, paths in tqdm(validloader):
                datas = datas.to(device)
                features, _ = model(datas)
                for feature, path in zip(features, paths):
                    t_i = path.split("/")[-2]
                    e_i = feature.detach().cpu().numpy()
                    # ...
                    embeddings_dict[t_i].append((path, e_i))

        genuine_scores, imposter_scores = matching_evaluation(embeddings_dict, all_imposter, False)
        fpr, tpr, thresholds, optimal_threshold = roc_threshold(imposter_scores[:,2].astype(np.float32), genuine_scores[:,2].astype(np.float32), save_dir, model_path.split('/')[-1])
        fprs.append(fpr)
        tprs.append(tpr)
        labels.append(model_path.split('/')[-1])
        print(model_path.split('/')[-1])


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Combine the ROC curves
    plt.figure()
    colour = ['r','g','b','y','m','tab:pink']
    i=0
    for fpr, tpr, model in zip(fprs, tprs, labels):
        i+=1
        plt.plot(fpr*100, tpr*100, label=model, color=colour[i%6])
    plt.xlim([0, 1])
    plt.ylim([0, 100])
    plt.title("Combined ROC")
    plt.ylabel("TAR or 1-FRR (%)")
    plt.xlabel("FAR (%)")
    plt.grid()
    plt.legend()
    roc_path = os.path.join(save_dir, 'combined_roc_TUFTS.png')
    plt.savefig(roc_path)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--ckpt", default="ir_101", help="model ckpt", type=str)
    parser.add_argument("--d", default="/research/prip-jainkush/AdaFace/data_root/data_test/IIITD", help="Input dir", type=str)
    # parser.add_argument("--d", default="/localscratch/groszste/FaceDatabases/CASIA-Webface_prip_aligned_mtcnn/val_2imgsperID_watermarked", help="Input dir", type=str)
    parser.add_argument('--first_imposter_only', action='store_true', help='Compute first imposters')
    parser.add_argument('--nonsymmetric', action='store_true', help='Compute entire similarity matrix, rather than just upper triangle')
    parser.add_argument('--save_dir', default='/research/prip-jainkush/AdaFace/experiments', help='save dir')
    parser.add_argument("--batch_size", default=512, help="batch size", type=int)
    args = parser.parse_args()
    epoch_score = []
    main(args)
    epoch_score = sorted(
    epoch_score, 
    key=lambda x: x[0]
    )
    print(epoch_score)