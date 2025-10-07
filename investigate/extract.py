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
