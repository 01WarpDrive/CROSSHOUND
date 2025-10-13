import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_log_file(filename):
    """
    解析日志文件，提取主体、行为、客体
    """
    events = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # 分割主体、行为、客体
            parts = line.split(' ', 2)
            if len(parts) >= 3:
                subject = parts[0]
                action = parts[1]
                object_ = parts[2]
                
                # 清理路径，提取有意义的节点名称
                subject_clean = clean_node_name(subject)
                object_clean = clean_node_name(object_)
                
                events.append({
                    'subject': subject_clean,
                    'action': action,
                    'object': object_clean,
                    'raw_subject': subject,
                    'raw_object': object_
                })
    
    return events

def clean_node_name(name):
    """
    清理节点名称，提取有意义的标识
    """
    # 如果是可执行文件路径，提取文件名
    if '\\' in name:
        # 提取最后一个反斜杠后的内容
        base_name = name.split('\\')[-1]
        if '.' in base_name and base_name not in ['$LogFile', 'pagefile.sys']:
            return base_name
        else:
            # 对于特殊文件，保留部分路径信息
            if name.startswith('\\Device\\HarddiskVolume1'):
                short_name = name.replace('\\Device\\HarddiskVolume1', 'C:')
                if len(short_name) > 30:
                    return '...' + short_name[-27:]
                return short_name
    return name

def build_directed_graph(events):
    """
    构建有向图
    """
    G = nx.DiGraph()
    edge_labels = {}
    node_colors = {}
    
    # 统计节点出现次数
    node_count = defaultdict(int)
    for event in events:
        node_count[event['subject']] += 1
        node_count[event['object']] += 1
    
    for event in events:
        subject = event['subject']
        object_ = event['object']
        action = event['action']
        
        # 添加节点
        G.add_node(subject)
        G.add_node(object_)
        
        # 添加边
        G.add_edge(subject, object_, action=action)
        
        # 记录边的标签（行为）
        edge_labels[(subject, object_)] = action
        
        # 根据节点类型设置颜色
        if 'python.exe' in subject.lower() or 'python.exe' in object_.lower():
            node_colors[subject] = 'red'
            node_colors[object_] = 'red'
        elif 'cmd.exe' in subject.lower() or 'cmd.exe' in object_.lower():
            node_colors[subject] = 'orange'
            node_colors[object_] = 'orange'
        elif 'powershell.exe' in subject.lower() or 'powershell.exe' in object_.lower():
            node_colors[subject] = 'purple'
            node_colors[object_] = 'purple'
        elif 'firefox.exe' in subject.lower() or 'firefox.exe' in object_.lower():
            node_colors[subject] = 'blue'
            node_colors[object_] = 'blue'
        elif 'svchost.exe' in subject.lower() or 'svchost.exe' in object_.lower():
            node_colors[subject] = 'green'
            node_colors[object_] = 'green'
        else:
            if subject not in node_colors:
                node_colors[subject] = 'lightgray'
            if object_ not in node_colors:
                node_colors[object_] = 'lightgray'
    
    return G, edge_labels, node_colors, node_count

def visualize_graph(G, edge_labels, node_colors, node_count, filename):
    """
    可视化有向图
    """
    plt.figure(figsize=(20, 15))
    
    # 使用spring布局
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 绘制节点
    node_sizes = [min(500 + node_count[node] * 50, 2000) for node in G.nodes()]
    node_color_list = [node_colors[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_color_list,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=1)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          alpha=0.7,
                          connectionstyle="arc3,rad=0.1")
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, 
                           font_size=8,
                           font_weight='bold')
    
    # 绘制边标签（只显示部分重要的边标签以避免混乱）
    important_actions = ['CREATE', 'MESSAGE', 'TERMINATE', 'START']
    filtered_edge_labels = {}
    for edge, label in edge_labels.items():
        if label in important_actions:
            filtered_edge_labels[edge] = label
    
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=filtered_edge_labels,
                                font_size=6,
                                font_color='red')
    
    plt.title('Attack Scenario Directed Graph\n(Red: Python, Orange: CMD, Purple: PowerShell, Blue: Firefox, Green: svchost)', 
              fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

def analyze_attack_patterns(events):
    """
    分析攻击模式
    """
    print("=== 攻击模式分析 ===")
    
    # 统计关键行为
    key_actions = defaultdict(int)
    python_activities = []
    network_activities = []
    process_creations = []
    
    for event in events:
        action = event['action']
        key_actions[action] += 1
        
        # Python相关活动
        if 'python' in event['subject'].lower() or 'python' in event['object'].lower():
            python_activities.append(event)
        
        # 网络活动
        if event['action'] in ['MESSAGE', 'START'] and '->' in event['raw_object']:
            network_activities.append(event)
        
        # 进程创建
        if event['action'] == 'CREATE' and '.exe' in event['object'].lower():
            process_creations.append(event)
    
    print(f"\n关键行为统计:")
    for action, count in sorted(key_actions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {action}: {count}次")
    
    print(f"\nPython相关活动: {len(python_activities)}次")
    print(f"网络连接活动: {len(network_activities)}次") 
    print(f"进程创建活动: {len(process_creations)}次")
    
    return python_activities, network_activities, process_creations

# 主执行流程
def main():
    root = './data/optc_day23/'
    # 解析日志文件
    print("正在解析日志文件...")
    events = parse_log_file(f'{root}optc_day23_investigate.txt')
    print(f"解析完成，共 {len(events)} 个事件")
    
    # 构建图
    print("正在构建有向图...")
    G, edge_labels, node_colors, node_count = build_directed_graph(events)
    
    print(f"图结构: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
    
    # 可视化
    print("正在生成可视化图形...")
    output_file = visualize_graph(G, edge_labels, node_colors, node_count, f'{root}attack_graph.png')
    print(f"图形已保存为: {output_file}")
    
    # # 分析攻击模式
    # python_acts, network_acts, process_creations = analyze_attack_patterns(events)
    
    # # 显示关键节点信息
    # print(f"\n=== 关键节点度中心性 ===")
    # degree_centrality = nx.degree_centrality(G)
    # top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # for node, centrality in top_nodes:
    #     print(f"  {node}: {centrality:.4f}")

if __name__ == "__main__":
    main()