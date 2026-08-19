import os

import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_NPU_PATH = os.path.dirname(torch_npu.__file__)
ASCEND_HOME = os.environ["ASCEND_HOME_PATH"]
BUILD_DIR = os.path.join(SRC_DIR, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

load(
    name="ffn_linear_custom",
    sources=[os.path.join(SRC_DIR, "ffn_linear_pytorch.cpp")],
    extra_include_paths=[
        os.path.join(TORCH_NPU_PATH, "include"),
        os.path.join(TORCH_NPU_PATH, "include/third_party/acl/inc"),
        os.path.join(ASCEND_HOME, "x86_64-linux/include"),
        os.path.join(ASCEND_HOME, "x86_64-linux/include/aclnn"),
        SRC_DIR,
    ],
    extra_ldflags=[
        f"-L{os.path.join(TORCH_NPU_PATH, 'lib')} -ltorch_npu",
        "-ldl",
    ],
    with_cuda=False,
    is_python_module=False,
    build_directory=BUILD_DIR,
    verbose=True,
)
print("OK")
