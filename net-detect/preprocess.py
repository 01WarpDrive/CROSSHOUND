"""
Provide high-performance satellite network traffic packet processing capabilities:
- It supports a throughput processing capacity of at least 100,000 data packets per second.
- Ensure that the data delay does not exceed 500ms.
"""


import json
import time
import math
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Generator, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import torch.nn.functional as F
from gensim.models import FastText
from dataclasses import dataclass
from collections import deque
from module.model import VAE
from module.config import DATASET_FILE_MAP

DEFAULT_BATCH_SIZE = 5000
TARGET_THROUGHPUT = 100000  # packets per second
MAX_LATENCY_MS = 500  # milliseconds


def Sentence_Construction(entry: Dict) -> List[str]:
    """构造句子短语"""
    return f"{entry['src_ip_port']} {entry['dest_ip_port']} {entry['type']}".split()


def batch_json_parse(lines: List[str]) -> List[Dict]:
    """批量解析JSON行"""
    return [json.loads(line) for line in lines]


def stream_data(file_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> Generator[List[Dict], None, None]:
    """
    流式读取JSON数据文件
    
    Args:
        file_path: 数据文件路径
        batch_size: 每批处理的行数
        
    Yields:
        每批解析后的JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        batch_lines = []
        for line in f:
            batch_lines.append(line.strip())
            if len(batch_lines) >= batch_size:
                yield batch_json_parse(batch_lines)
                batch_lines = []
        
        # 处理最后一批数据
        if batch_lines:
            yield batch_json_parse(batch_lines)


def process_batch_with_phrases(batch: List[Dict]) -> List[Dict]:
    """处理单个批次，添加phrase字段"""
    for event in batch:
        event['phrase'] = Sentence_Construction(event)
    return batch


def load_data_streaming(file_path: str, 
                       batch_size: int = DEFAULT_BATCH_SIZE, 
                       num_workers: int = 4) -> Generator[pd.DataFrame, None, None]:
    """
    流式加载和处理数据
    
    Args:
        file_path: 输入文件路径
        batch_size: 批次大小
        num_workers: 并行工作线程数
        
    Yields:
        处理后的DataFrame批次
    """
    total_records = 0
    batch_count = 0
    
    # 流式读取和处理数据
    for batch_idx, json_batch in enumerate(stream_data(file_path, batch_size)):
        batch_count += 1
        total_records += len(json_batch)
        
        # 并行处理phrase构造
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 将大批次分成更小的子批次进行并行处理
            sub_batch_size = max(1, len(json_batch) // num_workers)
            sub_batches = [json_batch[i:i + sub_batch_size] 
                          for i in range(0, len(json_batch), sub_batch_size)]
            
            # 并行处理子批次
            processed_sub_batches = list(executor.map(process_batch_with_phrases, sub_batches))
            
            # 合并处理后的数据
            processed_batch = list(chain.from_iterable(processed_sub_batches))
        
        # 转换为DataFrame并排序
        df_batch = pd.DataFrame(processed_batch)
        df_batch.sort_values('timestamp', inplace=True)
        
        # 输出进度
        if batch_idx % 100 == 0:
            print(f'Processed batch {batch_idx}, total records: {total_records}')
        
        yield df_batch
    
    print(f'Finish streaming loading. Processed {total_records} records in {batch_count} batches')



def construct_graph(df):
    """使用pandas向量化操作，最快的版本"""
    nodes = {}
    
    # 分组聚合，一次性处理所有相同的actor_id和object_id
    # 处理源IP
    src_groups = df.groupby('src_ip_port')['phrase'].agg(lambda x: list(x)).to_dict()
    for actor_id, phrase_list in src_groups.items():
        # 展平列表的列表
        flat_list = []
        for sublist in phrase_list:
            if isinstance(sublist, (list, tuple)):
                flat_list.extend(sublist)
            else:
                flat_list.append(sublist)
        nodes[actor_id] = flat_list
    
    # 处理目标IP
    dest_groups = df.groupby('dest_ip_port')['phrase'].agg(lambda x: list(x)).to_dict()
    for object_id, phrase_list in dest_groups.items():
        flat_list = []
        for sublist in phrase_list:
            if isinstance(sublist, (list, tuple)):
                flat_list.extend(sublist)
            else:
                flat_list.append(sublist)
        
        if object_id in nodes:
            nodes[object_id].extend(flat_list)
        else:
            nodes[object_id] = flat_list
    
    return nodes



def infer(document, w2vmodel):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(64)

    combined_embeddings = np.array(word_embeddings)
    output_embedding = torch.tensor(combined_embeddings, dtype=torch.float)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def Featurize(nodes: Dict, w2vmodel):
    """最快版本的特征提取函数"""
    features = []
    node_map_idx = {}
    
    # 预加载词汇表和词向量矩阵
    wv_vocab = set(w2vmodel.wv.key_to_index.keys())
    wv_vectors = w2vmodel.wv.vectors  # 所有词向量的矩阵
    
    # 构建词汇到索引的映射
    word_to_index = {word: idx for idx, word in enumerate(w2vmodel.wv.index_to_key)}
    
    for node, phrases in nodes.items():
        if len(phrases) <= 1:
            continue
            
        # 快速过滤和获取有效词的索引
        valid_indices = []
        for word in phrases:
            if word in word_to_index:
                valid_indices.append(word_to_index[word])
        
        if not valid_indices:
            features.append(np.zeros(64))
            node_map_idx[node] = len(features) - 1
            continue
        
        # 直接从词向量矩阵中批量获取嵌入
        word_embeddings = wv_vectors[valid_indices]
        
        # 使用numpy直接计算均值，避免torch转换开销
        doc_embedding = np.mean(word_embeddings, axis=0)
        
        features.append(doc_embedding)
        node_map_idx[node] = len(features) - 1
    
    return features, node_map_idx



def get_MSE(model, features, device):
    """最快版本的MSE计算函数"""
    # 快速转换为numpy数组
    if isinstance(features, list):
        features_arr = np.array(features, dtype=np.float32)
    else:
        features_arr = np.asarray(features, dtype=np.float32)
    
    # 直接创建tensor并移动到设备
    x = torch.as_tensor(features_arr, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
        
        # 使用更高效的MSE计算方式
        mse_loss = (x_recon - x).pow(2).sum(dim=1)
        
        # 直接返回numpy数组，避免tolist转换
        return mse_loss.cpu().numpy()


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    throughput: float = 0.0
    latency_ms: float = 0.0
    processing_time: float = 0.0
    packet_count: int = 0
    anomaly_count: int = 0
    timestamp: float = time.time()

class MetricEvaluation:
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.is_fitted = False


    def update_metrics(self, batch: pd.DataFrame, processing_time: float) -> PerformanceMetrics:
        """
        更新性能指标
        
        Args:
            batch: 处理的数据批次
            processing_time: 处理时间(秒)
            
        Returns:
            性能指标对象
        """
        packet_count = len(batch)
        throughput = packet_count / processing_time if processing_time > 0 else 0
        latency_ms = processing_time * 1000  # 转换为毫秒
        
        metrics = PerformanceMetrics(
            throughput=throughput,
            latency_ms=latency_ms,
            processing_time=processing_time,
            packet_count=packet_count
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_performance_targets(self) -> Tuple[bool, str]:
        """
        检查是否满足性能指标要求
        
        Returns:
            (是否满足要求, 详细消息)
        """
        if not self.metrics_history:
            return False, "No metrics data available"
        
        # 计算平均性能指标
        avg_throughput = np.mean([m.throughput for m in self.metrics_history])
        avg_latency = np.mean([m.latency_ms for m in self.metrics_history])
        total_packets = sum([m.packet_count for m in self.metrics_history])
        
        throughput_ok = avg_throughput >= TARGET_THROUGHPUT
        latency_ok = avg_latency <= MAX_LATENCY_MS
        
        message = (f"Throughput: {avg_throughput:,.0f} pps (Target: {TARGET_THROUGHPUT:,.0f} pps) - {'✓' if throughput_ok else '✗'}\n"
                  f"Latency: {avg_latency:.2f} ms (Target: ≤{MAX_LATENCY_MS} ms) - {'✓' if latency_ok else '✗'}\n"
                  f"Total Packets Processed: {total_packets:,.0f}")
        
        return throughput_ok and latency_ok, message


def main():
    """主函数 - 系统验证流程"""
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset

    dataset_path = f'./dataset/{dataset}/'
    TEST_FILE = f"{dataset_path}{DATASET_FILE_MAP[dataset]['test']}"
    FASTTEXT_PATH = DATASET_FILE_MAP[dataset]['FASTTEXT_PATH']
    VAE_PATH = DATASET_FILE_MAP[dataset]['VAE_PATH']

    w2vmodel = FastText.load(FASTTEXT_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()
    
    threshold = 91.10735867309342
    anomalies = set()
    metric_evaluate = MetricEvaluation()

    # stream preprocess and detect
    for i, df_batch in enumerate(load_data_streaming(TEST_FILE)):
        if df_batch.empty:
            continue
        batch_arrival_time = time.perf_counter()

        nodes = construct_graph(df_batch)
        features, test_node_index = Featurize(nodes, w2vmodel)
        node_ids = list(test_node_index)
        test_mse = get_MSE(model, features, device)

        completion_time = time.perf_counter()

        for id, mse in zip(node_ids, test_mse):
            if mse > threshold:
                anomalies.add(id)

        latency = completion_time - batch_arrival_time

        # 更新性能指标
        metrics = metric_evaluate.update_metrics(df_batch, latency)

        # 定期检查性能目标
        if i % 100 == 0 and i > 0:
            targets_met, message = metric_evaluate.check_performance_targets()
            print(f"Performance Check:\n{message}")
            
            if targets_met:
                print("All performance targets are being met!")
            else:
                print("Some performance targets are not met")

    # 最终性能报告
    targets_met, message = metric_evaluate.check_performance_targets()
    print(f"Final Performance Report:\n{message}")
    
    if targets_met:
        print("System validation PASSED - All targets achieved!")
    else:
        print("System validation FAILED - Targets not met")


if __name__ == "__main__":
    main()