#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import concurrent.futures
import os
from pathlib import Path

import pytest
import torch

import utils


TESTCASE_PATH = "./pt_files/"
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0


def execute_sfa(testcase_file):
    test_data = torch.load(testcase_file, map_location="cpu")
    utils.sfa_run_npu(test_data, device_id=DEVICE_ID, result_path=RESULT_PATH)


locals()["testcase_files"] = []
if os.path.isdir(TESTCASE_PATH):
    pt_files = [f for f in os.listdir(TESTCASE_PATH) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {TESTCASE_PATH}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(TESTCASE_PATH, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {TESTCASE_PATH}")


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", locals()["testcase_files"])
def test_sparse_flash_attention_batch(testcase_file):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_sfa, testcase_file)
        for done in concurrent.futures.as_completed([future]):
            try:
                done.result()
            except Exception as error:
                pytest.fail(f"当前用例线程执行失败: {error}")
