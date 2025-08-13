import os
import time
import torch
import warnings
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.loaddata import load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.config import build_args
warnings.filterwarnings('ignore')
from utils.utils import set_random_seed, create_optimizer
set_random_seed(0)


def draw_loss(loss_list, dataset_name):
    path = f'./figs/train_loss_{dataset_name}.png'
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    # plt.show()


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset
    if dataset_name == 'optc_day23':
        main_args.num_hidden = 64
        main_args.max_epoch = 100
        main_args.num_layers = 3
    elif dataset_name == 'optc_day24':
        main_args.num_hidden = 64
        main_args.max_epoch = 100
        main_args.num_layers = 2
    elif dataset_name == 'optc_day25':
        main_args.num_hidden = 64
        main_args.max_epoch = 100
        main_args.num_layers = 2
    metadata = load_metadata(dataset_name)
    main_args.n_dim = metadata['node_feature_dim']
    main_args.e_dim = metadata['edge_feature_dim']
    model = build_model(main_args)
    model = model.to(device)
    model.train()
    optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
    epoch_iter = tqdm(range(main_args.max_epoch))
    n_train = metadata['n_train']
    loss_list = []

    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i in range(n_train):
            g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
            loss = model(g)
            loss /= n_train
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            del g
        loss_list.append(epoch_loss)
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
    draw_loss(loss_list, dataset_name)

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
    if os.path.exists(save_dict_path):
        os.unlink(save_dict_path)
    return


if __name__ == '__main__':
    start_time = time.time()
    args = build_args()
    main(args)
    print(f'Finish: {time.time() - start_time} s')
