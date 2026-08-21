"""MlaPreprocess TestSpec 入口

ACLNN 模式 TestSpec, 从 impl/ 加载各模块。

三方对比模式:
  - golden:    高精度 CPU 输出 (cpu_mla_preprocess)
  - benchmark: 低精度 CPU 输出 (cpu_mla_preprocess_lowprec, 在 compare 中计算)
  - actual:    aclnn NPU 输出

C API 参数顺序 (aclnnMlaPreprocessGetWorkspaceSize):
  inputs: input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0,
          gamma1, beta1, quantScale1, quantOffset1, wuq, deScale1, bias1, gamma2,
          cos, sin, wuk, kvCache, kvCacheRope, slotmapping, ctkvScale, qNopeScale
  attr:   wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff,
          transposeWdq, transposeWuq, transposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount
  outputs: qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut
"""

import importlib.util
import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_impl(name):
    """从 impl/ 子目录加载模块 (spec loader 不注册包，不能用相对导入)"""
    path = os.path.join(_THIS_DIR, "impl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_mla_impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_impl("golden")
_inputs_mod = _load_impl("inputs")
_compare_mod = _load_impl("compare")

cpu_mla_preprocess = _golden_mod.cpu_mla_preprocess
cpu_mla_preprocess_lowprec = _golden_mod.cpu_mla_preprocess_lowprec
customize_inputs = _inputs_mod.customize_inputs
compare = _compare_mod.compare


class MlaPreprocessTestSpec:
    """MlaPreprocess 算子测试规范 — ACLNN 模式

    三方对比模式:
      - Actual:   aclnn NPU 输出 vs 高精度 golden
      - Benchmark: 低精度 CPU 输出 vs 高精度 golden
      - ratio = Actual 误差 / max(Benchmark 误差, error_thd)
    """

    # CPU golden 参考实现 — 参数签名与 aclnn 头文件一致
    golden = staticmethod(cpu_mla_preprocess)

    # 定制输入修正 — slotmapping/deScale/NZ 格式转换
    customize_inputs = staticmethod(customize_inputs)

    # 自定义精度比较 — 三方对比 (高精度CPU/低精度CPU/aclnn)
    compare = staticmethod(compare)

    # 精度标准
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
    }


__spec__ = {
    "aclnnMlaPreprocess": "MlaPreprocessTestSpec",
}
