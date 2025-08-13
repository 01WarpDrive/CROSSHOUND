import networkx as nx
import time
import pickle as pkl
import argparse
import numpy as np


def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    return prec, rec, fscore, FPR, TPR


def Get_Adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()
    
    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))
    
    return neighbors


def helper(MP, all_pids, GP, edges, mapp, distance=2):
    """FLASH OpTC evaluation method
    """
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = (GP - MP)
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, distance)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, distance)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR = calculate_metrics(TP, FP, FN, TN)
    print(f"Precision: {round(prec, 3)}, Recall: {round(rec, 3)}, Fscore: {round(fscore, 3)}")
    
    return TPL, FPL


def eval_optc_flash(test_g, test_ids, alarm_ids, gt_ids):
    edges = [[], []]
    for src, dst in test_g.edges():
        edges[0].append(src)
        edges[1].append(dst)
    alerts = helper(set(alarm_ids), set(test_ids), gt_ids, edges, test_ids)


def GetbackSubgraph(G, node, depth, sense):
    subgraph = set()
    if depth == 0:
        return subgraph

    score = {}
    for i in G.predecessors(node):
        score[i] = (G.nodes[i]['score'], G.out_degree(i) / (G.in_degree(i) + 1))
    
    new_score = sorted(score.items(), key=lambda d: (d[1][0],d[1][1]), reverse=True)[:10]
    node_list = [i[0] for i in new_score]
    for i in node_list:
        if i in sense:
            continue
        sense.add(i)
        subgraph.add(i)
        x = GetbackSubgraph(G, i,depth - 1,sense)
        subgraph |= x

    return subgraph


def GetforeSubgraph(G, node, depth, sense):
    subgraph = set()
    if depth == 0:
        return subgraph

    score = {}
    for i in G.successors(node):
        score[i] = (G.nodes[i]['score'], G.out_degree(i) / (G.in_degree(i) + 1))
    
    new_score = sorted(score.items(), key=lambda d:(d[1][0],d[1][1]), reverse=True)[:10]
    node_list = [i[0] for i in new_score]

    for i in node_list:
        if i in sense:
            continue
        sense.add(i)
        subgraph.add(i)
        x = GetforeSubgraph(G, i,depth - 1, sense)
        subgraph |= x

    return subgraph


def propagation(digraph, terminals, depth=3):
    subgraph_node = set(terminals)
    for node in terminals:
        sense = {node}
        local_subgraph = GetbackSubgraph(digraph, node, depth, sense)
        subgraph_node |= local_subgraph

        sense = {node}
        local_subgraph = GetforeSubgraph(digraph, node, depth, sense)
        subgraph_node |= local_subgraph
    subgraph = digraph.subgraph(list(subgraph_node)).copy()

    connected_graph = []
    for n in nx.weakly_connected_components(subgraph):
        g = subgraph.subgraph(n).copy()
        g.graph['score'] = np.sum([digraph.nodes[i]['score'] for i in g.nodes()])
        connected_graph.append(g)

    print(f'weak conn: {len(connected_graph)}')

    top_score = 50
    new_g = False
    for g in connected_graph:
        if g.graph['score'] > top_score:
            new_g = g
            top_score = g.graph['score']

    if not new_g:
        raise('No attack graph!')

    return new_g


def steiner_mst(graph, terminals):
    new_graph = propagation(graph, terminals)
    new_graph = new_graph.to_undirected()
    new_terminals = set()
    for n in terminals:
        if n in new_graph.nodes():
            new_terminals.add(n)
    complete_graph = nx.Graph()
    for u in new_terminals:
        dist = nx.single_source_shortest_path_length(new_graph, u)
        for v in new_terminals:
            if v != u:
                complete_graph.add_edge(u, v, weight=dist[v])

    mst_edges = nx.minimum_spanning_edges(complete_graph, data=True, algorithm='prim')
    steiner_edges = set()
    for u, v, d in mst_edges:
        path = nx.shortest_path(new_graph, u, v)
        steiner_edges.update(zip(path[:-1], path[1:]))

    return new_graph.edge_subgraph(steiner_edges)


def merge(host_g, alarm_idx): 
    for node in host_g.nodes():
        if node in alarm_idx:
            host_g.nodes[node]['score'] = 10
        else:
            host_g.nodes[node]['score'] = 0

    return host_g


def host_net_stp():
    start_time = time.time()
    print('host/net alarms stp')

    with open(HOST_GRAPH_PATH, 'rb') as f:
        graphs = pkl.load(f)
        host_g = nx.node_link_graph(graphs[0])
        print(f'graph nodes {host_g.number_of_nodes()}, edges {host_g.number_of_edges()}')
    with open(NODE_LIST_PATH, 'r') as file:
        node_list = file.read().split()
    with open(HOST_ALARM_PATH, 'r') as f:
        host_alarm_ids = set(f.read().split())
    with open(NET_ALARM_PATH, 'r') as f:
        net_alarm_ids = set(f.read().split())
    EntitySet = set(node_list)

    with open(GROUND_TRUTH_FILE, 'r') as file:
        gt_nodes = set(file.read().split())
    gt_nodes = set(x for x in gt_nodes if x in EntitySet)

    net_alarm_ids = set(x for x in net_alarm_ids if x in EntitySet)

    id_idx_map = {}
    for idx, id in enumerate(node_list):
        id_idx_map[id] = idx
    alarm_idx = set()
    alarm_nodes = host_alarm_ids | net_alarm_ids

    for id in alarm_nodes:
        try: 
            alarm_idx.add(id_idx_map[id])
        except:
            print(id)

    print('After STP')
    merge_g = merge(host_g, alarm_idx)
    st_tree = steiner_mst(merge_g, alarm_idx)

    print(f'st nodes: {len(st_tree.nodes())}')
    with open(OUTPUT_ALARM_PATH, 'w') as file:
        for node in st_tree.nodes():
            file.write(node_list[node])
            file.write('\n')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"finish: {elapsed_time:.4f} s")


def my_eval():
    with open(HOST_GRAPH_PATH, 'rb') as f:
        graphs = pkl.load(f)
        host_g = nx.node_link_graph(graphs[0])
    with open(NODE_LIST_PATH, 'r') as file:
        node_list = file.read().split()
    with open(OUTPUT_ALARM_PATH, 'r') as f:
        alarm_ids = set(f.read().split())

    EntitySet = set(node_list)
    with open(GROUND_TRUTH_FILE, 'r') as file:
        gt_nodes = set(file.read().split())
    gt_nodes = set(x for x in gt_nodes if x in EntitySet)

    eval_optc_flash(host_g, node_list, alarm_ids, gt_nodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23")
    args = parser.parse_args()
    dataset = args.dataset

    if 'optc' in dataset:
        HOST_GRAPH_PATH = f'../host-detect/data/{dataset}/test.pkl'
        NODE_LIST_PATH = f'../host-detect/data/{dataset}/node_list.txt'
        HOST_ALARM_PATH = f'../host-detect/data/{dataset}/alarm_list.txt'
        NET_ALARM_PATH = f'../net-detect/dataset/{dataset}-flow/net_alarms.txt'
        OUTPUT_ALARM_PATH = f'./data/{dataset}/alarms.txt'
        GROUND_TRUTH_FILE = f'../host-detect/data/{dataset}/{dataset}.txt'
        host_net_stp()
        my_eval()
