import json
import math
import time
import numpy as np
import pandas as pd
import torch
from gensim.models import FastText
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import argparse
from module.config import DATASET_FILE_MAP


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


def Sentence_Construction(entry):
    return f"{entry['src_ip_port']} {entry['dest_ip_port']} {entry['type']}".split()


def batch_json_parse(lines):
    return [json.loads(line) for line in lines]


def load_data(file_path, save_path=None, batch_size=10000, num_workers=4):
    print('Start loading')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data_batches = list(executor.map(batch_json_parse, batches))
    data = list(chain.from_iterable(data_batches))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for event in data:
            event['phrase'] = Sentence_Construction(event)
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    if save_path:
        df.to_parquet(save_path)
    print(f'Finish loading. Processed {len(df)} records')

    return df


def prepare_sentences(df):
    max_num = 100000
    nodes = {}
    for _, row in df.iterrows():
        for key in ['src_ip_port', 'dest_ip_port']:
            node_id = row[key]
            nodes.setdefault(node_id, []).extend(row['phrase'])
        if len(nodes) > max_num:
            break
    return list(nodes.values())


def train_FastText(events):
    """train FastText
    """
    phrases = prepare_sentences(events)

    print('Start training FastText')
    model = FastText(min_count=2, vector_size=64, workers=30, alpha=0.01, window=3, negative=3)
    model.build_vocab(phrases)
    model.train(phrases, epochs=50, total_examples=model.corpus_count)
    model.save(FASTTEXT_PATH)
    print(f'train model: {FASTTEXT_PATH}')


def infer(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(64)

    combined_embeddings = np.array(word_embeddings)
    output_embedding = torch.tensor(combined_embeddings, dtype=torch.float)

    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def Featurize(df):
    print('Start featuring')

    nodes = {} # {id of actor and object: }
    neimap = {}
    for _, row in df.iterrows():
        actor_id, object_id = row['src_ip_port'], row["dest_ip_port"]

        nodes.setdefault(actor_id, []).extend(row['phrase'])
        nodes.setdefault(object_id, []).extend(row['phrase'])

        neimap.setdefault(actor_id, set()).add(object_id)
        neimap.setdefault(object_id, set()).add(actor_id)

    features = []
    node_map_idx = {} # {node_id: index in features}

    for node, phrases in nodes.items():
        if len(phrases) > 1:
            features.append(infer(phrases))
            node_map_idx[node] = len(features) - 1

    print('finish featuring')

    return features, node_map_idx


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = PositionalEncoder(64)

    # load train data
    df = load_data(TRAIN_FILE)
    # train FastText
    train_FastText(df)

    end_time = time.time()
    print(f'Finish train FastText: {end_time - start_time}')
