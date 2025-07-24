import re
import json
import hashlib
from sql_alchemy_helper import SQL_Alchemy_Helper


SCHEMA_DIR = '../data/schema'
CONFIG_DIR = '../config/config.json'

database_config = json.load(open(CONFIG_DIR, 'r', encoding='utf-8'))
sql_alchemy_helper = SQL_Alchemy_Helper(database_config)

#name sanitization
def transfer_name(original_name):
    # 去除扩展名
    name = original_name.split('.')[0]
    
    # 替换非法字符为下划线
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # 去除连续下划线
    name = re.sub(r'_+', '_', name)
    
    # 去除首尾下划线
    if len(name) > 2:
        name = name.strip('_')
    
    # 转换为小写
    name = name.lower()
    
    # 确保不以数字开头
    if name[0].isdigit():
        name = 't_' + name
    
    # 处理超长名称，如果长度超过64个字符，截断并添加哈希后缀
    if len(name) > 64:
        prefix = name[:20].rstrip('_')
        hash_suffix = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        name = f"{prefix}_{hash_suffix}"
    
    return name