"""测试套件 C++ 编译器唯一定义点：容器内 BiSheng，不存在任何其他编译器路径。

构建与测试已全部收敛到算子门禁容器（docker/，见 docker/README.md）；本仓库测试
动态库（孪生执行器、量化核、宿主包装）统一由容器内 BiSheng 宿主模式编译。
非容器环境直跑本测试套件会在此处显式失败（请经 run.sh 自动包装进容器执行）；
特殊工具链路径仅支持经 CXX 环境变量显式指定，无自动回退链。
"""

import os
import shutil


def cxx() -> str:
    """返回编译测试动态库用的 C++ 编译器命令（容器内即 BiSheng）。"""
    explicit = os.environ.get("CXX")
    if explicit:
        return explicit
    found = shutil.which("bisheng")
    if found:
        return found
    raise RuntimeError(
        "未找到 BiSheng 编译器（bisheng）：请经 run.sh（自动包装进容器）执行本测试套件；"
        "特殊工具链请显式设置 CXX 环境变量。"
    )
