import json
import time
import argparse
import pickle as pkl


def match_optc(dataset):
    dataset_file_map = {
        'optc_day23': {
            'host_event': 'SysClient0201.systemia.com.json.txt',
            'net_data': 'test_conn_23_15-19.json'
        },
        'optc_day24': {
            'host_event': 'SysClient0501.systemia.com.json.txt',
            'net_data': 'test_conn_24_14-21.json'
        },
        'optc_day25': {
            'host_event': 'SysClient0051.systemia.com.json.txt',
            'net_data': 'test_conn_25_13-18.json'
        }
    }
    HOST_EVENT_PATH = f"../host-detect/data/{dataset}/{dataset_file_map[dataset]['host_event']}"
    NODE_NAME_PATH = f"../host-detect/data/{dataset}/names.json"
    NET_DATA_PATH = f"../net-detect/dataset/{dataset}-flow/{dataset_file_map[dataset]['net_data']}"
    MATCH_FLOW_PATH = f"./data/{dataset}/match_flows.pkl"

    # load host data
    with open(HOST_EVENT_PATH, 'r') as file:
        host_events = file.readlines()
    with open(NODE_NAME_PATH, 'r') as file:
        id_name_map = json.load(file)

    # get mapping relationship of network node and event timestamp
    hnode_timestamp_map = {} # {ip port: set(t1, ...)}
    for line in host_events:
        event = line.split()
        # net type node
        if event[3] == 'FLOW':
            id = event[2]
            t_host = int(event[5])
            try:
                node_name_list = id_name_map[id].split()
            except KeyError:
                continue
            src_ip = f'{node_name_list[0]} {node_name_list[1]}'
            dst_ip = f'{node_name_list[2]} {node_name_list[3]}'
            flow = (src_ip, dst_ip)
            if flow not in hnode_timestamp_map:
                hnode_timestamp_map[flow] = set()
            hnode_timestamp_map[flow].add(t_host)

    time_window = 500000
    match_flow = set()

    # load net data
    with open(NET_DATA_PATH, 'r') as f:
        lines = f.readlines()
    for line in lines:
        event = json.loads(line)
        t_net = int(float(event['timestamp']) * 1000)
        src_ip = event['src_ip_port']
        dst_ip = event['dest_ip_port']
        flow = (src_ip, dst_ip)
        if flow in hnode_timestamp_map:
            for t_host in hnode_timestamp_map[flow]:
                if abs(t_host - t_net) < time_window:
                    match_flow.add(flow)
                    break
    with open(MATCH_FLOW_PATH, 'wb') as f:
        pkl.dump(match_flow, f)


def match_lanl(dataset):
    HOST_EVENT_PATH = "../host-detect/data/lanl/test.txt"
    NET_DATA_PATH = "../net-detect/dataset/lanl-flow/test.json"
    MATCH_FLOW_PATH = f"./data/{dataset}/match_flows.pkl"
    # load host data
    with open(HOST_EVENT_PATH, 'r') as file:
        host_events = file.readlines()

    # get mapping relationship of network node and event timestamp
    hnode_timestamp_map = {} # {ip port: set(t1, ...)}
    for line in host_events:
        parts = line.strip().split('\t')
        src_id, dst_id, t_host = parts[0], parts[2], int(parts[5])
        flow = (src_id, dst_id)
        if flow not in hnode_timestamp_map:
            hnode_timestamp_map[flow] = set()
        hnode_timestamp_map[flow].add(t_host)

    time_window = 5
    match_flow = set()

    # load net data
    with open(NET_DATA_PATH, 'r') as f:
        lines = f.readlines()
    for line in lines:
        event = json.loads(line)
        t_net = int(event['timestamp'])
        src_ip = event['src_ip_port']
        dst_ip = event['dest_ip_port']
        flow = (src_ip, dst_ip)
        if flow in hnode_timestamp_map:
            for t_host in hnode_timestamp_map[flow]:
                if abs(t_host - t_net) < time_window:
                    match_flow.add(flow)
                    break
    with open(MATCH_FLOW_PATH, 'wb') as f:
        pkl.dump(match_flow, f)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23")
    args = parser.parse_args()
    dataset = args.dataset

    if 'optc' in dataset:
        match_optc(dataset)
    elif 'lanl' in dataset:
        match_lanl(dataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"finish: {elapsed_time:.4f} s")