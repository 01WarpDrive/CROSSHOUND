import argparse
import json
import os
from tqdm import tqdm
import networkx as nx
import pickle as pkl
from dateutil import parser as time_parser
import pytz
import time


node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0
metadata = {
    'optc_day23':{
            'train': ['benign_20-23Seq19_0201_.ecar-2019-12-07T19-16-05.788.json',
                    'benign_20-23Seq19_0201_.ecar-2019-12-07T22-06-33.589.json',
                    'benign_20-23Seq19_0201_.ecar-2019-12-08T01-57-30.012.json',
                    'benign_20-23Seq19_0201_.ecar-2019-12-08T05-46-21.658.json',
                    'benign_20-23Seq19_0201_.ecar-last.json'],
            'test': ['SysClient0201.systemia.com.json']
    },
    'optc_day24':{
            'train': ['benign_20-23Seq19_0501_.ecar-2019-11-15T03-10-00.546.json',
                    'benign_20-23Seq19_0501_.ecar-2019-11-15T05-59-37.208.json',
                    'benign_20-23Seq19_0501_.ecar-2019-11-15T09-43-35.856.json',
                    'benign_20-23Seq19_0501_.ecar-2019-11-15T13-29-59.064.json',
                    'benign_20-23Seq19_0501_.ecar-2019-11-15T17-22-42.923.json',
                    'benign_20-23Seq19_0501_.ecar-last.json'],
            'test': ['SysClient0501.systemia.com.json']
    },
    'optc_day25':{
            'train': ['benign_20-23Seq19_0051_.ecar-2019-12-07T16-15-43.163.json',
                    'benign_20-23Seq19_0051_.ecar-2019-12-07T18-18-31.331.json',
                    'benign_20-23Seq19_0051_.ecar-2019-12-07T21-31-30.259.json',
                    'benign_20-23Seq19_0051_.ecar-2019-12-08T00-56-58.175.json',
                    'benign_20-23Seq19_0051_.ecar-2019-12-08T04-30-36.852.json',
                    'benign_20-23Seq19_0051_.ecar-last.json'],
            'test': ['SysClient0051.systemia.com.json']
    }
}


def read_single_graph(dataset, malicious, path, test=False):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    print('converting {} ...'.format(path))
    path = '../data/{}/'.format(dataset) + path + '.txt'
    f = open(path, 'r')
    lines = []

    # for edge information
    for l in f.readlines():
        split_line = l.split('\t')
        src, src_type, dst, dst_type, edge_type, ts = split_line
        ts = int(ts)
        if not test:
            if src in malicious or dst in malicious:
                if src in malicious or dst in malicious:
                    continue

        # node type encoding
        if src_type not in node_type_dict:
            node_type_dict[src_type] = node_type_cnt
            node_type_cnt += 1
        if dst_type not in node_type_dict:
            node_type_dict[dst_type] = node_type_cnt
            node_type_cnt += 1

        # edge type encoding
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1

        # get edge information
        lines.append([src, dst, src_type, dst_type, edge_type, ts])

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []
    for l in lines:
        src, dst, src_type, dst_type, edge_type = l[:5]
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        # node encoding, add nodes
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        # add edges
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)

    return node_map, g, node_list


def ISO8601_to_UTC_millisecond(time_str):
    dt = time_parser.isoparse(time_str)
    dt_utc = dt.astimezone(pytz.UTC)
    timestamp_seconds = dt_utc.timestamp()
    timestamp_milliseconds = int(timestamp_seconds * 1000)

    return str(timestamp_milliseconds)


def preprocess_dataset_optc(dataset):
    id_nodetype_map = {}
    id_nodename_map = {}
    
    for file in os.listdir('../data/{}/'.format(dataset)):
        if 'json' in file and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file and not 'zeek' in file:
            print('reading {} ...'.format(file))
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            for line in tqdm(f):
                event = json.loads(line)
                type = event['object']
                if type not in ['PROCESS', 'FILE', 'FLOW']:
                    continue

                properties = event['properties']
                try:
                    actor_id = event['actorID']
                    id_nodetype_map[actor_id] = 'PROCESS'
                    object_id = event['objectID']
                    id_nodetype_map[object_id] = type
                    if type == 'FLOW':
                        id_nodename_map[actor_id] = properties['image_path']
                        id_nodename_map[object_id] = f"{properties['src_ip']} {properties['src_port']} {properties['dest_ip']} {properties['dest_port']} {properties['direction']}"
                    elif type == 'FILE':
                        id_nodename_map[actor_id] = properties['image_path']
                        id_nodename_map[object_id] = properties['file_path']
                    else:
                        id_nodename_map[actor_id] = properties['parent_image_path']
                        id_nodename_map[object_id] = properties['image_path']

                except KeyError:
                    continue

    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('../data/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))
            for line in tqdm(f):
                event = json.loads(line)
                try:
                    srcId = event['actorID']
                    srcType = id_nodetype_map[srcId]
                    dstId = event['objectID']
                    dstType = id_nodetype_map[dstId]
                    edgeType = event['action']
                    timestamp = ISO8601_to_UTC_millisecond(event['timestamp'])
                    this_edge = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId) + '\t' + str(
                            dstType) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                    fw.write(this_edge)
                except:
                    continue

            fw.close()
            f.close()
    
    if len(id_nodename_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)


def read_graphs(dataset):
    # load malicious enity ids
    malicious_entities = '../data/{}/{}.txt'.format(dataset, dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())

    # get mapping relationships, node/edge information
    preprocess_dataset_optc(dataset)

    # get graphs
    train_gs = []
    for file in metadata[dataset]['train']:
        node_map, train_g, _ = read_single_graph(dataset, malicious_entities, file, False)
        train_gs.append(train_g)

    test_gs = []
    # encode test node id
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        node_map, test_g, node_list = read_single_graph(dataset, malicious_entities, file, True)

        # save node id as the encoding order
        with open(f'../data/{dataset}/node_list.txt', 'w') as file:
            for id in node_list:
                file.write(id)
                file.write('\n')

        # # merge muti test data
        test_gs.append(test_g)
        for key in node_map:
            if key not in test_node_map:
                test_node_map[key] = node_map[key] + count_node

    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []

        # get the final malicious entities in test data
        for e in malicious_entities:
            if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                final_malicious_entities.append(test_node_map[e])
                if e in id_nodename_map:
                    malicious_names.append(id_nodename_map[e])
                    f.write('{}\t{}\n'.format(e, id_nodename_map[e]))
                else:
                    malicious_names.append(e)
                    f.write('{}\t{}\n'.format(e, e))
    else:
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(e)
                f.write('{}\t{}\n'.format(e, e))

    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))
    meta_file = f'../data/{dataset}/metadata.json'
    if os.path.exists(meta_file):
        os.remove(meta_file)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23")
    args = parser.parse_args()
    if args.dataset in ['optc_day23', 'optc_day24', 'optc_day25']:
        read_graphs(args.dataset)
    else:
        raise NotImplementedError
    print(f'Finish: {time.time() - start_time} s')
