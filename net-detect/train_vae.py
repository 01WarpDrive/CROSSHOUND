import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gensim.models import FastText
import argparse
from module.model import CustomDataset, VAE, loss_function
import pickle as pkl
from module.config import DATASET_FILE_MAP


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def train_VAE(train_X, num_epochs):
    data_loader = DataLoader(train_X, batch_size=128, shuffle=True, pin_memory=True)
    model = VAE()
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss_list = []

    for epoch in range(num_epochs):
        train_loss = 0
        for _, (x) in enumerate(data_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss = loss_function(x_recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
            train_loss += loss.item()
        epoch_loss = train_loss / len(data_loader.dataset)
        loss_list.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")

    torch.save(model.state_dict(), VAE_PATH)
    print('Training finish. ')


def get_MSE(model, features):
    if isinstance(features, list) and all(isinstance(arr, np.ndarray) for arr in features):
        features = np.stack(features)
    x = torch.from_numpy(np.asarray(features)).float().to(device)
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
        mse_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1).cpu().numpy()
    
    return mse_loss.tolist()


def get_threshold(model, features):
    validate_mse = get_MSE(model, features)
    threshold = np.percentile(validate_mse, 99.7)
    print('90: ',np.percentile(validate_mse,90))
    print('80: ',np.percentile(validate_mse,80))
    print('70: ',np.percentile(validate_mse,70))
    print('60: ',np.percentile(validate_mse,60))
    return threshold


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    TRAIN_FILE = f"{dataset_path}{DATASET_FILE_MAP[dataset]['train']}"
    TEST_FILE = f"{dataset_path}{DATASET_FILE_MAP[dataset]['test']}"
    FASTTEXT_PATH = DATASET_FILE_MAP[dataset]['FASTTEXT_PATH']
    VAE_PATH = DATASET_FILE_MAP[dataset]['VAE_PATH']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load train data
    with open(f"dataset/{dataset}/train_features.pkl", "rb") as f:
        features = pkl.load(f)
    w2vmodel = FastText.load(FASTTEXT_PATH)

    # train VAE
    train_X = CustomDataset(features)
    train_VAE(train_X, 10)

    # get threshold
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()
    threshold = get_threshold(model, features)
    print(f"threshold {threshold}")

    end_time = time.time()
    print(f'Finish: {end_time - start_time}')
