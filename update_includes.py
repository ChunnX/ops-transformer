#!/usr/bin/env python3
import os
import re

def update_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        content = content.replace('"err/ops_err.h"', '"framework/ops_err.h"')
        content = re.sub(r'"static/([^"]+)"', r'"framework/\1"', content)
        content = content.replace('"common/op_api_def.h"', '"framework/op_api_def.h"')
        content = content.replace('"common/tensor_util.h"', '"framework/tensor_util.h"')
        content = content.replace('"common/tensor_util.cpp"', '"framework/tensor_util.cpp"')
        content = content.replace('"tiling_sink/', '"op_tiling/')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
    return False

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_patterns = ['*.cpp', '*.h', '*.hpp']
    updated_files = []
    
    for pattern in file_patterns:
        for root, dirs, files in os.walk(root_dir):
            if '.git' in root or 'build' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith(('.cpp', '.h', '.hpp')):
                    filepath = os.path.join(root, file)
                    if update_file(filepath):
                        updated_files.append(filepath)
    
    print(f"Total source files updated: {len(updated_files)}")
    
    cmake_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'CMakeLists.txt':
                cmake_path = os.path.join(root, file)
                try:
                    with open(cmake_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'tiling_sink' in content:
                        content = content.replace('tiling_sink', 'op_tiling')
                        with open(cmake_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        cmake_files.append(cmake_path)
                except Exception as e:
                    print(f"Error updating CMakeLists.txt {cmake_path}: {e}")
    
    print(f"Total CMakeLists.txt files updated: {len(cmake_files)}")

if __name__ == '__main__':
    main()