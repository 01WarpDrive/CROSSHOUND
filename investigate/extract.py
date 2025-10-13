import orjson as json
from dateutil import parser as time_parser
import pytz


def ISO8601_to_UTC_millisecond(time_str):
    dt = time_parser.isoparse(time_str)
    dt_utc = dt.astimezone(pytz.UTC)
    timestamp_seconds = dt_utc.timestamp()
    timestamp_milliseconds = int(timestamp_seconds * 1000)

    return str(timestamp_milliseconds)


def remove_duplicates(lst):
    return list(dict.fromkeys(lst))


def write_list_to_file(lst, filename):
    """将列表每个元素作为一行写入文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 需要确保每个元素都是字符串，并添加换行符
        lines = [str(item) + '\n' for item in lst]
        f.writelines(lines)


def extract_key_log():
    metadata = {
        'optc_day23': 'SysClient0201.systemia.com.json',
        'optc_day24': 'SysClient0501.systemia.com.json',
        'optc_day25': 'SysClient0051.systemia.com.json'

    }

    dataset = "optc_day25"
    ALARM_PATH = f'./data/{dataset}/alarms.txt'
    LOG_PATH = f'../host-detect/data/{dataset}/{metadata[dataset]}'
    OUTPUT_PATH = f'./data/{dataset}/{dataset}_investigate.txt'

    format_strings = {
        'PROCESS': "{parent_image_path} {action} {image_path}", # {command_line}
        'FILE': "{image_path} {action} {file_path}",
        'FLOW': "{image_path} {action} {src_ip}:{src_port}->{dest_ip}:{dest_port}",
        'MODULE': "{image_path} {action} {module_path}"
    }
    default_format = "{image_path} {action} {module_path}"

    with open(ALARM_PATH, 'r') as file:
        node_set = set(file.read().split())

    related_logs = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            event = json.loads(line)
            type = event['object']
            if type not in ['PROCESS', 'FILE', 'FLOW', 'MODULE']:
                continue

            actor_id = event['actorID']
            object_id = event['objectID']
            if actor_id in node_set and object_id in node_set:
                event['timestamp'] = ISO8601_to_UTC_millisecond(event['timestamp'])
                related_logs.append(event)

    related_logs = sorted(related_logs, key=lambda x: x['timestamp'])

    extracted_logs = []
    for entry in related_logs:
        action = entry["action"]
        properties = entry['properties']
        object_type = entry['object']
        try:
            format_str = format_strings.get(object_type, default_format)
            key_log = format_str.format(action=action, **properties)
        except KeyError:
            continue
        extracted_logs.append(key_log)

    extracted_logs = remove_duplicates(extracted_logs)
    write_list_to_file(extracted_logs, OUTPUT_PATH)


def extract_raw_log():
    metadata = {
        'optc_day23': 'SysClient0201.systemia.com.json',
        'optc_day24': 'SysClient0501.systemia.com.json',
        'optc_day25': 'SysClient0051.systemia.com.json'

    }

    dataset = "optc_day25"
    ALARM_PATH = f'./data/{dataset}/alarms.txt'
    LOG_PATH = f'../host-detect/data/{dataset}/{metadata[dataset]}'
    OUTPUT_PATH = f'./data/{dataset}/{dataset}_investigate_raw.txt'

    format_strings = {
        'PROCESS': "{parent_image_path} {action} {image_path}", # {command_line}
        'FILE': "{image_path} {action} {file_path}",
        'FLOW': "{image_path} {action} {src_ip}:{src_port}->{dest_ip}:{dest_port}",
        'MODULE': "{image_path} {action} {module_path}"
    }
    default_format = "{image_path} {action} {module_path}"

    with open(ALARM_PATH, 'r') as file:
        node_set = set(file.read().split())

    related_logs = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            event = json.loads(line)
            type = event['object']
            if type not in ['PROCESS', 'FILE', 'FLOW', 'MODULE']:
                continue

            actor_id = event['actorID']
            object_id = event['objectID']
            if actor_id in node_set and object_id in node_set:
                event['timestamp'] = ISO8601_to_UTC_millisecond(event['timestamp'])
                related_logs.append(event)

    related_logs = sorted(related_logs, key=lambda x: x['timestamp'])

    extracted_logs = []
    tmp = set()
    for entry in related_logs:
        action = entry["action"]
        properties = entry['properties']
        object_type = entry['object']
        try:
            format_str = format_strings.get(object_type, default_format)
            key_log = format_str.format(action=action, **properties)
        except KeyError:
            continue
        if key_log not in tmp:
            tmp.add(key_log)
            extracted_logs.append(str(entry))

    extracted_logs = remove_duplicates(extracted_logs)
    write_list_to_file(extracted_logs, OUTPUT_PATH)


def search():
    metadata = {
        'optc_day23': 'SysClient0201.systemia.com.json',
        'optc_day24': 'SysClient0501.systemia.com.json',
        'optc_day25': 'SysClient0051.systemia.com.json'

    }
    dataset = "optc_day25"
    LOG_PATH = f'../host-detect/data/{dataset}/{metadata[dataset]}'

    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if 'i' in line and ' get_gui' in line:
                print(line)
                return
    
    print('no result')


import chardet

def convert_to_utf8(file_path):
    """
    检测文件编码并转换为UTF-8
    """
    try:
        # 读取文件二进制内容
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        
        # 检测编码
        detected_encoding = chardet.detect(raw_data)['encoding']
        print(f"检测到编码: {detected_encoding}")
        
        # 如果已经是UTF-8，直接返回
        if detected_encoding and detected_encoding.lower() == 'utf-8':
            print("文件已经是UTF-8编码")
            return True
        
        # 解码并重新编码为UTF-8
        if detected_encoding:
            text_content = raw_data.decode(detected_encoding, errors='replace')
        else:
            # 如果无法检测编码，尝试常见编码
            for encoding in ['gbk', 'gb2312', 'latin-1', 'iso-8859-1']:
                try:
                    text_content = raw_data.decode(encoding)
                    print(f"使用备用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # 如果所有编码都失败，使用替换错误处理
                text_content = raw_data.decode('utf-8', errors='replace')
        
        # 写入UTF-8编码的文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
        
        print(f"成功将文件转换为UTF-8编码")
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

# 使用示例
convert_to_utf8('./data/redteam.txt')