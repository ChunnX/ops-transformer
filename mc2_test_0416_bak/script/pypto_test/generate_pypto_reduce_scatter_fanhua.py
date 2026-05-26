#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
import time

def BytesOf(dtype):
    """获取数据类型的字节数"""
    bytes_dict = {
        'int32': 4,
        'float16': 2,
        'float32': 4,
        'bfloat16': 2
    }
    return bytes_dict.get(dtype, 4)  

def generate_tile_shapes(row, col, worldSize, is_row_divisible=False, is_col_divisible=False):
    while True:
        # 随机生成tile_row_shape和tile_col_shape
        tile_row_shape = random.randint(1, row) 
        tile_col_shape = random.randint(1, col)
        
        # 如果要求row可整除tile_row_shape，则调整tile_row_shape
        if is_row_divisible and row % tile_row_shape != 0:
            # 调整tile_row_shape为row的因数
            factors = []
            for i in range(1, row + 1):
                if row % i == 0:
                    factors.append(i)
            if factors:
                tile_row_shape = random.choice(factors)
        
        # 如果要求col可整除tile_col_shape，则调整tile_col_shape
        if is_col_divisible and col % tile_col_shape != 0:
            factors = []
            for i in range(1, col + 1):
                if col % i == 0:
                    factors.append(i)
            if factors:
                tile_col_shape = random.choice(factors)
        
        # 计算tileNum
        tileNum = (row // tile_row_shape) * (col // tile_col_shape)
        
        # 检查约束条件
        if tileNum * worldSize <= 1024:
            return tile_row_shape, tile_col_shape

def generate_shmem_all_gather_test_cases(total_cases=200):
    """
    生成指定数量的ShmemReduceScatter测试用例
    """
    # 测试用例列表
    test_cases = []
    
    # 支持的数据类型
    dtypes = ['int32', 'float16', 'float32', 'bfloat16']
    
    # 设备数量：2-8卡全覆盖,
    rank_sizes = [ 2, 3, 4, 5, 6, 7, 8]
    
    # format选项,1-ND,0-NZ
    formats = ["ND"]
    
    # 不同数据类型对应的input_shape_range范围
    shape_ranges = {
        'int32': [[-3, 3], [-1, 1], [-10, 10], [-20, 20], [-5, 5], [-7, 7]],
        'float16': [[0, 0.001], [0.001, 0.01], [0.01, 0.1], [0, 1], [1, 1], [-0.001, 0], [-0.01, -0.001],
                 [-0.1, -0.01], [-6.1e-05, 6.1e-05], [-0.01, 0.01], [-0.1, 0.1], [-1, 1], [-100, 100], [-1000, 1000], 
                 [-10000, 10000], [-20000, 20000], [-3.40282e+38, 3.40282e+38]],
        'bfloat16': [[0, 0.001], [0.001, 0.01], [0.01, 0.1], [0, 1], [1, 1], [-0.001, 0], [-0.01, -0.001],
                 [-0.1, -0.01], [-6.1e-05, 6.1e-05], [-0.01, 0.01], [-0.1, 0.1], [-1, 1], [-100, 100], [-1000, 1000], 
                 [-10000, 10000], [-20000, 20000], [-65504, 65504]],
        'float32': [[0, 0.001], [0.001, 0.01], [0.01, 0.1], [0, 1], [1, 1], [-0.001, 0], [-0.01, -0.001],
                [-0.1, -0.01], [-6.1e-05, 6.1e-05], [-0.01, 0.01], [-0.1, 0.1], [-1, 1], [-100, 100], [-1000, 1000], 
                [-10000, 10000], [-20000, 20000], [-65504, 65504]]
    }   

    # 生成测试用例
    for i in range(total_cases):
        case_id = i + 1
        
        # 随机选择数据类型
        dtype = random.choice(dtypes)
        # 随机选择rankSize (2-8)
        rank_size = random.choice(rank_sizes)

        #signal预留内存大小worldsize*1024*8*4
        signal_size = rank_size * 1024 * 8 * 4

        # win区、shape存储大小
        win_size = 1024 * 1024 * 200

        # 随机生成row
        row = random.randint(1, 8192)
         
        # 确保row能被card整除
        if row % rank_size != 0:
            row = row - (row % rank_size)

        # col最大值小于 (win_size - signal_size)/rank_size/row/BytesOf(dtype) 
        col_max = (win_size - signal_size) // rank_size // row // BytesOf(dtype)  
		
        # 随机生成col 
        col = random.randint(0, col_max)

        # 随机选择format
        format_val = random.choice(formats)
        # 如果format=1,NZ，则确保col*BytesOf(dtype) % 32 == 0
        if format_val == "NZ":
            # 计算当前col值需要满足的条件
            bytes_per_element = BytesOf(dtype)
            # 确保col * bytes_per_element 能被32整除
            while (col * bytes_per_element) % 32 != 0:
                col += 1
      
        # 随机生成input_shape,支持2维tensor
        input_shape =f"[{row}, {col}]" 

        quarter = total_cases // 2
        if i < quarter:
            is_row_divisible = True
            is_col_divisible = True
        else:
            is_row_divisible = True
            is_col_divisible = False
        
        # 随机生成row和col的切块大小
        tile_row_shape, tile_col_shape = generate_tile_shapes(row//rank_size, col, rank_size, is_row_divisible, is_col_divisible)
               
        # input_shape_range根据dtype设置不同的范围
        range_pair = random.choice(shape_ranges[dtype])        
        input_shape_range = str(range_pair)
            
        # 构造用例名称
        case_name = f'shmem_reduce_scatter_{case_id}_{dtype}_{row}_{col}_{rank_size}_{tile_row_shape}_{tile_col_shape}_{format_val}'

        test_cases.append({
            'case_name': case_name,
            'input_shape':input_shape,
            'input_dtype': f"{dtype}",
            'input_format': format_val,
            'input_datarange': input_shape_range,
            'output_shape':f"[{row // rank_size}, {col}]",
            'output_dtype': f"{dtype}",
            'output_format': format_val,
            'view_shape': "[]",
            'tile_shape': "[]",
            'rankSize': rank_size,
            'tile_row_shape': tile_row_shape,
            'tile_col_shape': tile_col_shape
        })

    # 写入CSV文件
    fieldnames = ['case_name', 'input_shape', 'input_dtype', 'input_format',  'input_datarange', 'output_shape', 'output_dtype', 'output_format', 'view_shape', 'tile_shape', 'rankSize', 'tile_row_shape', 'tile_col_shape']
    
    timestamp = time.time()
    readable_time = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp))
    
    with open('open_shmem_reduce_scatter_newtile-'+readable_time+'_onlyND.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_cases)
    
    print(f"成功生成 {len(test_cases)} 个ShmemReducescatter测试用例")

if __name__ == "__main__":
    # 生成N个测试用例
    generate_shmem_all_gather_test_cases(200)
