import json
import time
import math
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Generator, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import chain, islice
import torch.nn.functional as F
from gensim.models import FastText
from dataclasses import dataclass
from collections import deque, defaultdict
from module.model import VAE
from module.config import DATASET_FILE_MAP

# 性能参数
DEFAULT_BATCH_SIZE = 10000  # 增大批次大小
TARGET_THROUGHPUT = 100000
MAX_LATENCY_MS = 500
PREFETCH_FACTOR = 2  # 预取因子

def batch_json_parse_optimized(lines: List[str]) -> List[Dict]:
    """优化版JSON解析"""
    return [json.loads(line) for line in lines]

def stream_data_optimized(file_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> Generator[List[Dict], None, None]:
    """优化版流式读取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            batch_lines = list(islice(f, batch_size))
            if not batch_lines:
                break
            yield batch_json_parse_optimized(batch_lines)

def Sentence_Construction_optimized(entry: Dict) -> List[str]:
    """优化版句子构造"""
    src = entry.get('src_ip_port', '')
    dest = entry.get('dest_ip_port', '')
    type_ = entry.get('type', '')
    return f"{src} {dest} {type_}".split()

def process_batch_parallel(batch: List[Dict]) -> List[Dict]:
    """并行处理批次"""
    for event in batch:
        event['phrase'] = Sentence_Construction_optimized(event)
    return batch

def load_data_streaming_optimized(file_path: str, batch_size: int = DEFAULT_BATCH_SIZE, num_workers: int = 8) -> Generator[pd.DataFrame, None, None]:
    """优化版流式加载"""
    total_records = 0
    
    for batch_idx, json_batch in enumerate(stream_data_optimized(file_path, batch_size)):
        total_records += len(json_batch)
        
        # 使用进程池并行处理（CPU密集型任务）
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            sub_batch_size = max(1, len(json_batch) // (num_workers * 2))
            sub_batches = [json_batch[i:i + sub_batch_size] for i in range(0, len(json_batch), sub_batch_size)]
            processed_sub_batches = list(executor.map(process_batch_parallel, sub_batches))
            processed_batch = list(chain.from_iterable(processed_sub_batches))
        
        # 使用pandas的优化操作
        df_batch = pd.DataFrame(processed_batch)
        if not df_batch.empty:
            df_batch.sort_values('timestamp', inplace=True, kind='mergesort')
        
        if batch_idx % 50 == 0:
            print(f'Processed batch {batch_idx}, records: {total_records:,}')
        
        yield df_batch

def construct_graph_optimized(df):
    """极致优化的图构建"""
    nodes = defaultdict(list)
    
    # 使用向量化操作
    src_phrases = df[['src_ip_port', 'phrase']].values
    dest_phrases = df[['dest_ip_port', 'phrase']].values
    
    # 处理源IP
    for src, phrase in src_phrases:
        if isinstance(phrase, (list, tuple)):
            nodes[src].extend(phrase)
        else:
            nodes[src].append(phrase)
    
    # 处理目标IP
    for dest, phrase in dest_phrases:
        if isinstance(phrase, (list, tuple)):
            nodes[dest].extend(phrase)
        else:
            nodes[dest].append(phrase)
    
    return dict(nodes)

def Featurize_optimized(nodes: Dict, w2vmodel, wv_vectors, word_to_index):
    """预加载数据的特征提取"""
    features = []
    node_map_idx = {}
    
    for node, phrases in nodes.items():
        if len(phrases) <= 1:
            continue
            
        # 使用集合推导式加速
        valid_indices = [word_to_index[word] for word in phrases if word in word_to_index]
        
        if not valid_indices:
            features.append(np.zeros(64, dtype=np.float32))
            node_map_idx[node] = len(features) - 1
            continue
        
        # 批量获取并计算均值
        word_embeddings = wv_vectors[valid_indices]
        doc_embedding = np.mean(word_embeddings, axis=0, dtype=np.float32)
        features.append(doc_embedding)
        node_map_idx[node] = len(features) - 1
    
    return features, node_map_idx

def get_MSE_optimized(model, features, device):
    """极致优化的MSE计算"""
    features_arr = np.array(features, dtype=np.float32, copy=False)
    x = torch.from_numpy(features_arr).to(device, non_blocking=True)
    
    with torch.inference_mode():
        x_recon, mu, logvar = model(x)
        mse_loss = torch.sum((x_recon - x) ** 2, dim=1)
        return mse_loss.cpu().numpy()

class PipelineOptimizer:
    """流水线优化器"""
    def __init__(self, w2vmodel, model, device, threshold):
        self.w2vmodel = w2vmodel
        self.model = model
        self.device = device
        self.threshold = threshold
        
        # 预加载数据
        self.wv_vectors = w2vmodel.wv.vectors.astype(np.float32)
        self.word_to_index = {word: idx for idx, word in enumerate(w2vmodel.wv.index_to_key)}
        
        # 预热模型
        self._warmup_model()
    
    def _warmup_model(self):
        """预热模型"""
        dummy_input = torch.randn(32, 64, device=self.device)
        with torch.inference_mode():
            self.model(dummy_input)
    
    def process_batch(self, df_batch):
        """处理单个批次"""
        batch_arrival_time = time.perf_counter()
        
        # 并行处理图构建和特征提取
        nodes = construct_graph_optimized(df_batch)
        features, test_node_index = Featurize_optimized(nodes, self.w2vmodel, self.wv_vectors, self.word_to_index)
        
        if features:
            test_mse = get_MSE_optimized(self.model, features, self.device)
            anomalies = [node for node, mse in zip(test_node_index.keys(), test_mse) if mse > self.threshold]
        else:
            anomalies = []
        
        completion_time = time.perf_counter()
        latency = completion_time - batch_arrival_time
        
        return anomalies, latency, len(df_batch)


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



def main_optimized():
    """优化版主函数"""
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset

    dataset_path = f'./dataset/{dataset}/'
    TEST_FILE = f"{dataset_path}{DATASET_FILE_MAP[dataset]['test']}"
    FASTTEXT_PATH = DATASET_FILE_MAP[dataset]['FASTTEXT_PATH']
    VAE_PATH = DATASET_FILE_MAP[dataset]['VAE_PATH']

    # 加载模型
    w2vmodel = FastText.load(FASTTEXT_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()
    
    threshold = 91.10735867309342
    anomalies = set()
    metric_evaluate = MetricEvaluation()
    
    # 创建流水线优化器
    pipeline = PipelineOptimizer(w2vmodel, model, device, threshold)

    # 使用更大的批次和并行处理
    batch_generator = load_data_streaming_optimized(TEST_FILE, batch_size=20000, num_workers=8)

    for i, df_batch in enumerate(batch_generator):
        if df_batch.empty:
            continue
        
        # 处理批次
        batch_anomalies, latency, packet_count = pipeline.process_batch(df_batch)
        anomalies.update(batch_anomalies)
        
        # 更新指标
        metrics = metric_evaluate.update_metrics(df_batch, latency)

        # 性能监控
        if i % 20 == 0:
            targets_met, message = metric_evaluate.check_performance_targets()
            print(f"Batch {i}: {message}")
            
            if not targets_met and i > 100:
                print("Warning: Performance targets not met, adjusting parameters...")
                # 这里可以添加动态调整逻辑

    # 最终报告
    targets_met, message = metric_evaluate.check_performance_targets()
    print(f"Final Performance:\n{message}")
    print(f"Total anomalies detected: {len(anomalies)}")

if __name__ == "__main__":
    main_optimized()