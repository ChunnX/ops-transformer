#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob

def modify_catlass_headers(catlass_dir):
    """
    Modify catlass header files to change #include "catlass/xxx" to #include "xxx"
    """
    if not os.path.exists(catlass_dir):
        print(f"Error: Directory {catlass_dir} does not exist")
        return 1
    
    # Find all .hpp files in catlass directory
    pattern = os.path.join(catlass_dir, "**/*.hpp")
    hpp_files = glob.glob(pattern, recursive=True)
    
    modified_count = 0
    for hpp_file in hpp_files:
        try:
            with open(hpp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace #include "catlass/xxx" with #include "xxx"
            new_content = content.replace('#include "catlass/', '#include "')
            
            if new_content != content:
                with open(hpp_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                modified_count += 1
                print(f"Modified: {hpp_file}")
        except Exception as e:
            print(f"Error processing {hpp_file}: {e}")
    
    print(f"Total modified {modified_count} files")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python modify_catlass_headers.py <catlass_include_dir>")
        sys.exit(1)
    
    catlass_dir = sys.argv[1]
    sys.exit(modify_catlass_headers(catlass_dir))