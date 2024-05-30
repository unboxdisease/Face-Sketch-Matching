import os
import glob
from tqdm import tqdm
from collections import defaultdict
import pickle
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import shutil
from matplotlib.ticker import ScalarFormatter

def get_rank_n(cmc_scores, rank=1):
    # assumes scores are already sorted largest to small
    correct = 0 
    total = 0
    for negatives, positives in cmc_scores:
        max_positive = max(positives[:rank])
        min_negative = min(negatives[:rank])
        # if max_positive == min_negative:
        #     print(f'correct mate has same score (s={max_positive}) as highest imposter match (s={min_negative})... in this case we will count it as a miss')
        if max_positive > min_negative:
            correct += 1
        total += 1
    # print('rank {} accuracy: {}%'.format(rank, correct/total * 100))
    return correct/total * 100

def plot_cmc(ranks=[1,5,10, 20, 100, 150, 500], recognition_rates=[95,96,97, 99, 99.5, 99.55, 99.8], save_path='plots/cmc.png'):
    plt.figure()
    plt.plot(ranks, recognition_rates)
    plt.grid(True)
    plt.title('Closed set search')
    plt.xlabel('Rank')
    plt.xticks([1,5,10, 20, 100, 150, 500])
    plt.xscale('log')
    plt.xlim([1, 501])
    plt.ylim([0,100])
    plt.ylabel('Identification Rate (%)')
    # plt.legend()
    plt.savefig(save_path)

def cmc_curve(probe_subjs, matches, failure_dir='./failure_cases'):
    if not os.path.isdir('/'.join(failure_dir.split('/')[:-1])+'/failure_cases'):
        os.makedirs('/'.join(failure_dir.split('/')[:-1])+'/failure_cases')

    cmc_scores = []
    count_no_match = 0

    for subj in tqdm(probe_subjs):
        negative_scores = np.array(matches[subj][1])[:, 0].astype(np.float32)
        negative_scores.sort()
        negative_scores = negative_scores[::-1]

        if len(matches[subj][0]) <= 0:
            continue
        positive_scores = np.array(matches[subj][0])[:, 0].astype(np.float32)
        positive_pair = (matches[subj][0][0][1], matches[subj][0][0][2])

        positive_scores.sort()
        positive_scores = positive_scores[::-1]
        cmc_scores.append((negative_scores, positive_scores))

        f1 = matches[subj][0][0][1].split('/')[-1].split('.')[0]
        f2 = matches[subj][0][0][2].split('/')[-1].split('.')[0]
        f = f'{f1}_{f2}.jpg'
        if positive_scores[0] <= negative_scores[0]:
            img_path = '/'.join(failure_dir.split('/')[:-1]) + f'/failure_cases/{f}'
            assert len(positive_pair) >= 2
            # img1 = cv2.imread(positive_pair[0])
            # img2 = cv2.imread(positive_pair[1])
            # img = np.hstack((img1, img2))
            # cv2.imwrite(img_path, img)

    ranks = [1, 5, 10, 15, 20, 30, 35, 50, 100, 150, 500]
    rank_rates = [get_rank_n(cmc_scores, rank=r) for r in ranks]

    print("Rank 1 identification: {:.2f}%".format(rank_rates[0] * 100))
    print("Rank 5 identification: {:.2f}%".format(rank_rates[1] * 100))
    print("Rank 10 identification: {:.2f}%".format(rank_rates[2] * 100))
    print("Rank 15 identification: {:.2f}%".format(rank_rates[3] * 100))
    print("Rank 20 identification: {:.2f}%".format(rank_rates[4] * 100))
    print("Rank 30 identification: {:.2f}%".format(rank_rates[5] * 100))
    print("Rank 35 identification: {:.2f}%".format(rank_rates[6] * 100))
    print("Rank 50 identification: {:.2f}%".format(rank_rates[7] * 100))
    print("Rank 100 identification: {:.2f}%".format(rank_rates[8] * 100))
    print("Rank 150 identification: {:.2f}%".format(rank_rates[9] * 100))
    print("Rank 500 identification: {:.2f}%".format(rank_rates[10] * 100))

    print(f"Number of probes with mates missing from score file: {count_no_match}")

    return cmc_scores, rank_rates, ranks

def get_gallery_dataset(path):
    if 'casia_vs' in path:
        dataset = 'casia_vs'
    elif "IITD" in path:
        dataset = 'IITD'
    elif "NTU-CP-v1" in path:
        dataset = 'NTU-CP-v1'
    elif 'adultcrossdb' in path:
        dataset = 'adultcrossdb'
    elif 'childcrossdb_2_3' in path:
        dataset = 'childcrossdb_2_3'
    else:
        raise Exception('invalid gallery dataset path')
    return dataset

def load(score_file, exclude_paths=set(), exclude_IDs=set()):
    lines = [line.strip() for line in open(score_file).readlines()]
    print('num lines to parse: ', len(lines))
    # lines.sort()
    matches = {}
    probe_subjs = set()
    excluded_paths = set()
    excluded_IDs = set()

    for line in tqdm(lines):
        # print(line)
        # sys.exit()
        probe_path, gallery_path, score = line.split(',')
        score = float(score[1:-1])
        probe_id = probe_path.split('/')[-2]
        gallery_id = gallery_path.split('/')[-2]
        if '/'.join(probe_path.split('/')[-2:]).split('.')[0] in exclude_paths:
            excluded_paths.add(probe_path)
            continue

        if gallery_id in exclude_IDs :
            excluded_IDs.add(gallery_id)
            continue

        if probe_path not in matches:
            probe_subjs.add(probe_path)
            matches[probe_path] = ([],[])

        if gallery_id == probe_id :
            matches[probe_path][0].append((score, probe_path, gallery_path))
        else:
            matches[probe_path][1].append((score, probe_path, gallery_path))

    # count_missing = 0
    # count_total = 0
    # for probe_path, items in matches.items():
    #     if len(items[0]) <= 0:
    #         count_missing += 1
    #         print(probe_path)
    #     count_total += 1
    # print(count_total)
    # print(count_missing)
    # sys.exit()

    print("number of paths excluded: ", len(excluded_paths))
    print("expected number paths excluded: ", len(exclude_paths))
    print("number of IDs excluded: ", len(excluded_IDs))
    print("expected number IDs excluded: ", len(exclude_IDs))

    return matches

def closed_set(subjs, matches, save_path="closed-set.png", failure_dir=None):
    cmc_scores, rates, ranks = cmc_curve(subjs, matches, failure_dir)
    plot_cmc(ranks, rates, save_path=save_path)
    return ranks, rates

def open_set(mated_probes, nonmated_probes, matches, save_path="open-set.png", rank=1):
    # compute FNIR for mated probes
    thresholds = np.linspace(0, 1.0, num=1000)
    # print('thresholds: ', thresholds)
    misses = [0] * len(thresholds)
    FNIRs = []
    FPIRs = []
    total = 0
    RANK = rank
    for probe_subj in mated_probes:
        scores = matches[probe_subj][0] + matches[probe_subj][1]
        scores.sort(key=lambda t: float(t[0]))
        scores = scores[::-1]

        # probe_dataset = probe_subj.split('/')[-4]
        probe_id = probe_subj.split('/')[-2]
        # for item in scores[:RANK]:
        #     assert probe_id == item[1].split('/')[-2]
        relevant = [i[2].split('/')[-2] for i in scores[:RANK]]
        if probe_id not in relevant:
            # print(f'probe not in rank: {RANK} list - ', probe_subj, scores[0][2], scores[0][0])
            for thresh_idx, threshold in enumerate(thresholds):
                misses[thresh_idx] += 1
                # (probe_id, relevant[0])
        else:
            #also check score is above threshold
            max_score = 0
            for score, _, gallery_id in scores[:RANK]:
                gallery_id = gallery_id.split('/')[-2]
                if probe_id == gallery_id:
                    if score > max_score:
                        max_id = gallery_id
                        max_score = score

            for thresh_idx, threshold in enumerate(thresholds):
                if max_score < threshold:
                    # if max_score <= 0:
                    #     print('less than zero: ', max_score)
                    misses[thresh_idx] += 1
                    # (probe_id, max_id)

        total += 1
    FNIRs = (np.array(misses)/total).tolist()

    # FNIRs = [float("{:.4f}".format(FNIR)) for FNIR in FNIRs]

    # # Find the threshold where FNIR is closest to 2%
    # target_FNR = 0.02
    # closest_idx = np.argmin(np.abs(np.array(FNIRs) - target_FNR))
    # closest_threshold = thresholds[closest_idx]
    # print(f"Closest threshold for FNIR=2%: {closest_threshold:.6f}")

    # # Calculate TPIR at the closest threshold
    # total_mated = len(mated_probes)
    # true_positives = total_mated - misses[closest_idx]
    # TPIR = true_positives / total_mated
    # print(f"TPIR at FNIR=2%: {TPIR:.2%}")

    # compute FPIR with non mated probes
    total = 0
    misses = [0] * len(thresholds)

    for probe_subj in nonmated_probes:
        scores = matches[probe_subj][0] + matches[probe_subj][1]
        scores.sort(key=lambda t: t[0])
        scores = scores[::-1]

        for thresh_idx, threshold in enumerate(thresholds):
            if scores[0][0] >= threshold:
                misses[thresh_idx] += 1
        total += 1

    FPIRs = (np.array(misses)/total).tolist()

    # FPIRs = [float("{:.4f}".format(FPIR)) for FPIR in FPIRs]
    target_fp = 0.02
    closest_fp_index = min(range(len(FPIRs)), key=lambda i: abs(FPIRs[i] - target_fp))
    fnir_at_2_percent_fpir = FNIRs[closest_fp_index]

    print("FNIR at FPIR = 2%: {:.2f}%".format(fnir_at_2_percent_fpir * 100))

    # for FPIR, threshold in zip(FPIRs, thresholds):
    #     print("FPIR = {:.2f}%, thr: {:.2f}".format(FPIR*100, threshold))
    
    plt.figure()
    plt.plot(FPIRs, FNIRs)
    plt.grid(True)
    plt.title('DET Curve')
    plt.xlabel('False Positive Identification Rate')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([0.0, 1])
    # plt.ylim([0.0, 1])
    plt.ylabel('False Negative Identification Rate')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter()) # 2 decimal places
    plt.gca().yaxis.set_major_formatter(ScalarFormatter()) # 2 decimal places
    plt.gca().ticklabel_format(axis='x', style='plain')
    plt.gca().ticklabel_format(axis='y', style='plain')
    # plt.xticks([0.001, 0.01, 0.1, 0.2, 0.4, 1.0])
    # plt.yticks([0.001, 0.01, 0.1, 0.2, 0.4, 1.0])
    plt.savefig(save_path)

    np.save(save_path.replace('.png', 'fpir_comb.npy'), FPIRs)
    np.save(save_path.replace('.png', 'fnir_comb.npy'), FNIRs)

def main(args):
    f = args.f
    closed_set_path = os.path.join('/'.join(f.split('/')[:-1]),'closed_set.png')
    open_set_path = os.path.join('/'.join(f.split('/')[:-1]),'open_set.png')
    # failure_dir = os.path.join('/'.join(f.split('/')[:-1]),'failures')
    # if not os.path.isdir(failure_dir):
    #     os.makedirs(failure_dir)
    failure_dir = None
    openset = True

    np.random.seed(12)
    exclude_paths = set()

    print('loading scores...')
    matches = load(f, exclude_paths=exclude_paths)
    subjs = list(matches.keys())
    print('number of prob images: ', len(subjs))
    IDs = defaultdict(list)
    identities = set()
    for subj in subjs:
        ID = subj.split('/')[-2]
        identities.add(ID)
        IDs[ID].append(subj)
    identities = list(identities)
    print('number of prob IDs: ', len(IDs))

    print('computing closed set identification...')
    closed_set(subjs, matches, save_path=closed_set_path, failure_dir=f)

    if openset:
        print('\ncomputing open set identification.....')
        print('splitting subjs into half mated and half nonmated...')
        n = len(identities)
        random.shuffle(identities)
        mated_probeIDs = identities[:n//2]
        nonmated_probeIDs = identities[n//2:]

        mated_probes = []
        nonmated_probes = []

        for probeID in mated_probeIDs:
            mated_probes += IDs[probeID]
        for probeID in nonmated_probeIDs:
            nonmated_probes += IDs[probeID]

        print('number of mated probe IDs: ', len(mated_probeIDs))
        print('number of nonmated probe IDs: ', len(nonmated_probeIDs))

        print('number of mated probe samples: ', len(mated_probes))
        print('number of nonmated probe samples: ', len(nonmated_probes))      

        print('reloading scores while dropping nonmated probes from gallery...')
        matches = load(f, exclude_paths=exclude_paths, exclude_IDs=nonmated_probeIDs)

        open_set(mated_probes, nonmated_probes, matches, save_path=open_set_path, rank=1)

def search(f, openset):
    # f = args.f
    closed_set_path = os.path.join('/'.join(f.split('/')[:-1]),'closed_set.png')
    open_set_path = os.path.join('/'.join(f.split('/')[:-1]),'open_set.png')
    # failure_dir = os.path.join('/'.join(f.split('/')[:-1]),'failures')
    openset = True
    failure_dir = None

    np.random.seed(12)

    print('loading scores...')
    matches = load(f)
    subjs = list(matches.keys())
    print('number of prob images: ', len(subjs))
    IDs = defaultdict(list)
    identities = set()
    for subj in subjs:
        ID = subj.split('/')[-2]
        identities.add(ID)
        IDs[ID].append(subj)
    identities = list(identities)
    print('number of prob IDs: ', len(IDs))

    print('computing closed set identification...')
    closed_set(subjs, matches, save_path=closed_set_path, failure_dir=failure_dir)

    if openset:
        print('\ncomputing open set identification.....')
        print('splitting subjs into half mated and half nonmated...')
        n = len(identities)
        random.shuffle(identities)
        mated_probeIDs = identities[:n//2]
        nonmated_probeIDs = identities[n//2:]

        mated_probes = []
        nonmated_probes = []

        for probeID in mated_probeIDs:
            mated_probes += IDs[probeID]
        for probeID in nonmated_probeIDs:
            nonmated_probes += IDs[probeID]

        print('number of mated probe IDs: ', len(mated_probeIDs))
        print('number of nonmated probe IDs: ', len(nonmated_probeIDs))

        print('number of mated probe samples: ', len(mated_probes))
        print('number of nonmated probe samples: ', len(nonmated_probes))      

        print('reloading scores while dropping nonmated probes from gallery...')
        matches = load(f, exclude_IDs=nonmated_probeIDs)

        open_set(mated_probes, nonmated_probes, matches, save_path=open_set_path, rank=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default='/research/prip-jainkush/AdaFace/Results/Openset_viewed/base/Viewed_scores.txt', type=str, metavar='FILENAME', help='Score file')
    parser.add_argument('--openset', action='store_true', help='Compute open-set performance')    
    args = parser.parse_args()
    main(args)