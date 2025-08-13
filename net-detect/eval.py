import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from module.model import VAE
import pickle as pkl
from module.config import DATASET_FILE_MAP


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def get_MSE(model, features):
    if isinstance(features, list) and all(isinstance(arr, np.ndarray) for arr in features):
        features = np.stack(features)
    x = torch.from_numpy(np.asarray(features)).float().to(device)
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
        mse_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1).cpu().numpy()
    
    return mse_loss.tolist()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    VAE_PATH = DATASET_FILE_MAP[dataset]['VAE_PATH']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()

    # get from ./train_vae.py
    threshold = 91.10735867309342

    # load test data
    with open(f"dataset/{dataset}/test_features.pkl", "rb") as f:
        features = pkl.load(f)
    with open(f"dataset/{dataset}/test_node_map_idx.pkl", "rb") as f:
        node_map_idx = pkl.load(f)
    node_ids = list(node_map_idx)

    # detection
    test_mse = get_MSE(model, features)
    anomalies = set()
    for id, mse in zip(node_ids, test_mse):
        if mse > threshold:
            anomalies.add(id)

    # optc host&net matching for convenience
    if 'optc' in dataset:
        EACRBRO_FILE = f"{dataset_path}{DATASET_FILE_MAP[dataset]['ecarbro']}"
        ip_objectID_map = {}
        with open(EACRBRO_FILE, 'r') as file:
            for line in file:
                event = json.loads(line)
                src_ip_port = f"{event['properties']['src_ip']} {event['properties']['src_port']}"
                dst_ip_port= f"{event['properties']['dest_ip']} {event['properties']['dest_port']}"

                for ip_port in [src_ip_port, dst_ip_port]:
                    if ip_port not in ip_objectID_map:
                        ip_objectID_map[ip_port] = {event['objectID']}
                    else:
                        ip_objectID_map[ip_port].add(event['objectID'])

        anomaly_ecar = set()
        for ip_port in anomalies:
            if ip_port in ip_objectID_map:
                anomaly_ecar |= ip_objectID_map[ip_port]

        print(f"total enenties: {len(node_ids)} \nnet anomalies: {len(anomalies)} \nhost related anomalies num: {len(anomaly_ecar)}")
        with open(OUTPUT_FILE, 'w') as file:
            for id in anomaly_ecar:
                file.write(id)
                file.write('\n')
    else:
        print(f"total enenties: {len(node_ids)} \nnet anomalies: {len(anomalies)}")
        with open(OUTPUT_FILE, 'w') as file:
            for id in anomalies:
                file.write(id)
                file.write('\n')

    end_time = time.time()
    print(f'Finish eval: {end_time - start_time}')
