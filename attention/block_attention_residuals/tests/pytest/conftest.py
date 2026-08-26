# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""pytest 配置：注册自定义 mark。"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "ci: CI 必跑用例")
    config.addinivalue_line("markers", "extended: 扩展 NPU 扫参（pick10 / 历史失败）")
    config.addinivalue_line("markers", "resident: TK-RESIDENT 驻留分支")
    config.addinivalue_line("markers", "sweep: 正向 shape 扫描（如 H 递增）")
    config.addinivalue_line(
        "markers", "grid_tn: 泛化精度 N=1..12 × T 采样[1,16384]（H=7168，data_compare 门控）"
    )
