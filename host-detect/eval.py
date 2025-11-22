import torch
import warnings
import numpy as np
from utils.loaddata import load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.utils import set_random_seed
from model.eval import evaluate_entity_level_using_knn
from utils.config import build_args
from utils.utils import set_random_seed
import time
warnings.filterwarnings('ignore')
set_random_seed(0)


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    print(f"GPU/CPU: {device}")
    device = torch.device(device)
    dataset_name = main_args.dataset
    if dataset_name == 'optc_day23':
        main_args.num_hidden = 64
        main_args.num_layers = 3
    elif dataset_name == 'optc_day24':
        main_args.num_hidden = 64
        main_args.num_layers = 2
    elif dataset_name == 'optc_day25':
        main_args.num_hidden = 64
        main_args.num_layers = 2
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    metadata = load_metadata(dataset_name)
    main_args.n_dim = metadata['node_feature_dim']
    main_args.e_dim = metadata['edge_feature_dim']
    model = build_model(main_args)
    model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
    model = model.to(device)
    model.eval()
    malicious, _ = metadata['malicious']
    n_train = metadata['n_train']
    n_test = metadata['n_test']

    with torch.no_grad():
        x_train = []
        for i in range(n_train):
            g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
            x_train.append(model.embed(g).cpu().numpy())
            del g
        x_train = np.concatenate(x_train, axis=0)
        skip_benign = 0
        x_test = []
        for i in range(n_test):
            g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
            if i != n_test - 1:
                skip_benign += g.number_of_nodes()
            x_test.append(model.embed(g).cpu().numpy())
            del g
        x_test = np.concatenate(x_test, axis=0)
        n = x_test.shape[0]
        y_test = np.zeros(n)
        y_test[malicious] = 1.0
        malicious_dict = {}
        for i, m in enumerate(malicious):
            malicious_dict[m] = i
        test_idx = []
        for i in range(x_test.shape[0]):
            if i >= skip_benign or y_test[i] == 1.0:
                test_idx.append(i)
        result_x_test = x_test[test_idx]
        result_y_test = y_test[test_idx]

        evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test, result_y_test)


if __name__ == '__main__':
    start_time = time.time()
    args = build_args()
    main(args)
    print(f'Finish: {time.time() - start_time} s')
