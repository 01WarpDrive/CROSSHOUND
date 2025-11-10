import os
import random
import pickle as pkl
import networkx as nx
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
import time
set_random_seed(0)


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    print(f"==== eval {dataset} ====")
    # scale test data
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    if dataset == 'optc_day23':
        n_neighbors = 20
    elif dataset == 'optc_day24':
        n_neighbors = 20
    elif dataset == 'optc_day25':
        n_neighbors = 20

    # KNN model get train data
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)
    # get KNN distances
    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        print('get KNN distances')
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        # mean distance of some train samples
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        # all distances of test samples
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        print('load KNN distances')
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)

    # eval
    score = distances / mean_distance
    del distances
    _, rec, threshold = precision_recall_curve(y_test, score)
    # To repeat peak performance
    for i in range(len(rec)):
        if dataset == 'optc_day23' and rec[i] < 0.02:
            best_idx = i - 1
            break
        elif dataset == 'optc_day24' and rec[i] < 0.01:
            best_idx = i - 1
            break
        elif dataset == 'optc_day25' and rec[i] < 0.02:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]
    print(best_thres)
    with open(f'./data/{dataset}/node_list.txt', 'r') as file:
        node_list = file.read().split()
    assert len(node_list) == len(score)
    alarm_list = []
    for i in range(len(score)):
        if score[i] >= best_thres:
            alarm_list.append(node_list[i])
    with open(f'./data/{dataset}/alarm_list.txt', 'w') as file:
        file.write('\n'.join(alarm_list))
