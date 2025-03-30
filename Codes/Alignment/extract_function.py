import json
import os
import re
from tree_sitter import Language, Parser

# 加载C语言模块
Language.build_library(
    'build/my-language.so',
    [
        './tree-sitter-c'
    ]
)

C_LANGUAGE = Language('build/my-language.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)
JSON_PATH = None

# 提取节点信息
def extract_node_information(node, code):
    try:
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        # 将源代码按行进行拆分
        code_lines = code.split('\n')
        # 如果起始行和结束行在同一行
        if start_row == end_row:
            extracted_code = code_lines[start_row][start_col:end_col]
        else:
            # 提取起始行到结束行中的内容
            extracted_code = code_lines[start_row][start_col:]
            for i in range(start_row + 1, end_row):
                extracted_code += code_lines[i] + '\n'
            extracted_code += code_lines[end_row][:end_col]
        return extracted_code
    except AttributeError as e:
        return ''

# 提取代码信息
def extract_code_information(node, code):
    result = []
    if node is None:
        return
    stack = [node]
    while stack:
        node = stack.pop()
        if node.type == 'function_definition':
            result.append({
                '函数名': traverse_children(node, code),
                '函数': extract_node_information(node, code)
            })
        for child in reversed(node.children):
            stack.append(child)
    return result

# 深度优先遍历节点的全部孩子节点
def traverse_children(node, code):
    if node is None:
        return ''
    if node.type == 'identifier':
        return extract_node_information(node, code)
    for child in node.children:
        result = traverse_children(child, code)
        if result:
            return result
    return ''

# 查找文件夹中的.c文件
def get_c_files(folder):
    c_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if re.search(r'\.c$', file):
                c_files.append(os.path.join(root, file))
    return c_files

# 处理文件夹
def pipeline(folder_path):
    c_files = get_c_files(folder_path)
    functions = []
    count = 0
    for c_file in c_files:
        count += 1
        # print(str(count) + ": " + c_file)
        encoders = ['GB2312', 'utf-8', 'gb18030', 'gbk', 'ISO-8859-1']
        for encoder in encoders:
            try:
                with open(c_file, 'r', encoding=encoder) as file:
                    code = file.read()
                    tree = parser.parse(bytes(code, encoder))
                    root_node = tree.root_node
                    functions.extend(extract_code_information(root_node, code))
                    break
            except UnicodeDecodeError as e:
                if encoder == encoders[-1]:
                    print('UnicodeDecodeError!', c_file)
                continue
    global JSON_PATH
    JSON_PATH = os.path.join(folder_path, 'functions.json')
    with open(JSON_PATH, 'w', encoding='utf8') as json_file:
        json.dump(functions, json_file, indent=4, ensure_ascii=False)

def search_func(func_name):
    global JSON_PATH
    with open(JSON_PATH, 'r', encoding='utf8') as file:
        data = json.load(file)
        for obj in data:
            if obj['函数名'] in func_name:
                return obj['函数']
        return ''

def search_functions(func_names):
    functions = []
    func_names = str(func_names)
    
    if func_names == '':
        pass
    else:
        if ',' in func_names:
            func_names_list = func_names.split(',')
            for func_name in func_names_list:
                functions.append(search_func(func_name))
        else:
            functions = search_func(func_names)
    
    return functions
    