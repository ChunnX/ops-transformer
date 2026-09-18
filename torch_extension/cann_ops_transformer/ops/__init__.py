# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lazy op registry for ``cann_ops_transformer.ops``.

导入本包只做目录扫描与 entry point 收集（毫秒级），不触发任何算子模块的
import / schema 注册 / JIT 编译。算子只在属性被访问时按需加载（PEP 562），
避免"使用任意单个算子却编译全部算子"的问题。

如需旧行为（一次性加载全部算子，例如要求 ``torch.ops.cann_ops_transformer``
全量注册），可显式调用 :func:`load_all`。
"""

import ast
import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

# legacy 旧接口名 -> 新算子模块相对路径（相对于本包）
_legacy_map = {
    "attention_to_ffn": "mc2.attention_to_ffn_v2",
    "causal_conv1d_fn": "mamba.causal_conv1d",
    "causal_conv1d_update": "mamba.causal_conv1d",
    "comm_context": "mc2.common",
    "compressor": "attention.compressor",
    "deep_ep": "mc2.moe_distribute_dispatch",
    "dense_lightning_indexer_kl_loss_grad": "attention.dense_lightning_indexer_kl_loss_grad",
    "dense_lightning_indexer_softmax_lse": "attention.dense_lightning_indexer_softmax_lse_v2",
    "elastic_buffer": "mc2.common",
    "flash_attn": "attention.flash_attn",
    "flash_mla_with_kvcache": "attention.flash_mla_with_kvcache",
    "flash_attn_grad": "attention.flash_attn_grad",
    "generic_block_sparse_attention": "attention.generic_block_sparse_attention",
    "ffn_to_attention": "mc2.ffn_to_attention_v2",
    "grouped_matmul_activation_quant": "gmm.grouped_matmul_activation_quant",
    "indexer_quant_cache": "attention.indexer_quant_cache",
    "inplace_partial_rotary_mul": "posembedding.inplace_partial_rotary_mul",
    "inplace_partial_rotary_mul_backward": "posembedding.inplace_partial_rotary_mul_grad",
    "kv_compress_epilog": "attention.kv_compress_epilog",
    "kv_quant_sparse_flash_attention": "attention.kv_quant_sparse_flash_attention_v2",
    "lightning_indexer": "attention.lightning_indexer_v2",
    "lightning_indexer_kl_loss": "attention.lightning_indexer_kl_loss",
    "mega_moe": "mc2.mega_moe",
    "quant_reduce_scatter": "mc2.quant_reduce_scatter",
    "quant_all_reduce": "mc2.quant_all_reduce",
    "mhc_post": "mhc.mhc_post",
    "mhc_post_backward": "mhc.mhc_post_backward",
    "mhc_pre_sinkhorn": "mhc.mhc_pre_sinkhorn",
    "mhc_pre_sinkhorn_backward": "mhc.mhc_pre_sinkhorn_backward",
    "mixed_quant_sparse_flash_mla": "attention.mixed_quant_sparse_flash_mla",
    "moe_finalize_routing": "moe.moe_finalize_routing_v2",
    "moe_init_routing": "moe.moe_init_routing_v4",
    "moe_re_routing": "moe.moe_re_routing_v2",
    "moe_token_permute": "moe.moe_token_permute",
    "msa_index_score": "attention.msa_index_score",
    "qkv_rms_norm_rope_cache_with_k_scale": "posembedding.qkv_rms_norm_rope_cache_with_k_scale",
    "quant_compressor": "attention.quant_compressor",
    "quant_flash_attn": "attention.quant_flash_attn",
    "quant_lightning_indexer": "attention.quant_lightning_indexer_v2",
    "quant_sparse_flash_mla": "attention.quant_sparse_flash_mla",
    "scatter_pa_kv_cache_with_k_scale": "attention.scatter_pa_kv_cache_with_k_scale",
    "sparse_flash_mla": "attention.sparse_flash_mla",
    "sparse_flash_mla_grad": "attention.sparse_flash_mla_grad",
    "sparse_flash_mla_softmax_l1_norm": "attention.sparse_flash_mla_softmax_l1_norm",
    "sparse_lightning_indexer_kl_loss_grad": "attention.sparse_lightning_indexer_kl_loss_grad",
    "generic_block_sparse_attention_grad": "attention.generic_block_sparse_attention_grad",
    "stem_oam_prep_paged_kv": "attention.stem_oam_prep_paged_kv",
    "stem_oam_prep_varlen_q": "attention.stem_oam_prep_varlen_q",
    "und_gen_qkv_rms_norm_rope_cache": "posembedding.und_gen_qkv_rms_norm_rope_cache",
    "fused_causal_conv1d": "attention.fused_causal_conv1d",
    "fused_causal_conv1d_": "attention.inplace_fused_causal_conv1d",
    "block_sparse_attention": "attention.block_sparse_attention",
    "block_attention_residuals": "mhc.block_attention_residuals",
    "block_attention_residuals_backward": "mhc.block_attention_residuals_grad",
    "apply_rotary_pos_emb": "posembedding.apply_rotary_pos_emb",
    "apply_rotary_pos_emb_grad": "posembedding.apply_rotary_pos_emb_grad",
    "mla_prolog": "attention.mla_prolog_v3",
}


def _discover_ops_from_entry_points():
    ops = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        eps = (
            eps.select(group="cann_ops_transformer.ops")
            if hasattr(eps, "select")
            else eps.get("cann_ops_transformer.ops", [])
        )
        for ep in eps:
            ops[ep.name] = ep.value
    except (ImportError, RuntimeError, AttributeError):
        pass
    return ops


def _parse_exported_names(init_path):
    """静态解析算子 ``__init__.py`` 顶层的 ``__all__`` 字面量。

    仅读取文件内容做 AST 解析，不执行模块代码。解析失败时返回空列表，
    对应算子的额外导出退化为不可懒寻址（不影响主名访问）。
    """
    try:
        with open(init_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return []
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value = node.value
        if value is None:
            continue
        try:
            names = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(names, (list, tuple)):
            return [n for n in names if isinstance(n, str)]
        return []
    return []


def _parse_convert_funcs(gc_path):
    """静态解析 graph_convert 模块中可导出的 ``convert_*`` 函数名。

    收集模块顶层定义与 ``if _TORCHAIR_AVAILABLE:`` 分支内定义的函数名
    （即安装了 torchair 的环境下可访问的名字，与旧版在该环境下的运行时
    导出行为一致）。仅出现在 ``else:`` 分支（无 torchair fallback stub）
    而未在 if 分支定义的名字不返回，避免在有 torchair 环境下误导。
    仅做 AST 解析，不执行模块代码。
    """
    try:
        with open(gc_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return set()

    def _collect(body):
        names = set()
        for node in body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("convert_"):
                names.add(node.name)
        return names

    names = _collect(tree.body)
    for node in tree.body:
        if isinstance(node, ast.If):
            names |= _collect(node.body)
    return names


def _discover_ops_from_dir():
    """扫描本包目录，返回 (算子注册表, 额外导出索引, 类别名集合, converter函数索引)。

    算子注册表: {算子名: 模块路径}
    额外导出索引: {__all__ 中的额外导出名: 算子名}
    类别名集合: ops/ 下的一级子目录名（namespace 包），用于属性访问兜底
    converter函数索引: {convert_函数名: 算子名}（静态解析 graph_convert 模块）
    """
    ops = {}
    exports = {}
    categories = set()
    convert_funcs = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_name = __name__

    _skip = frozenset(("graph_convert",))
    for entry in sorted(os.listdir(current_dir)):
        if entry in _skip or entry.startswith((".", "_", "__")):
            continue
        entry_path = os.path.join(current_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        categories.add(entry)
        for sub in sorted(os.listdir(entry_path)):
            if sub in _skip or sub.startswith((".", "_", "__")):
                continue
            sub_path = os.path.join(entry_path, sub)
            sub_init = os.path.join(sub_path, "__init__.py")
            if os.path.isdir(sub_path) and os.path.isfile(sub_init) and sub not in ops:
                ops[sub] = "%s.%s.%s" % (package_name, entry, sub)
                gc_path = os.path.join(sub_path, "graph_convert_%s.py" % sub)
                if os.path.isfile(gc_path):
                    for func_name in _parse_convert_funcs(gc_path):
                        if func_name not in convert_funcs:
                            convert_funcs[func_name] = sub
                for extra in _parse_exported_names(sub_init):
                    if extra != sub and extra not in exports:
                        exports[extra] = sub

    return ops, exports, categories, convert_funcs


_ep_ops = _discover_ops_from_entry_points()
_dir_ops, _dir_exports, _category_names, _convert_funcs = _discover_ops_from_dir()

_op_registry = dict(_dir_ops)
_op_registry.update(_ep_ops)

# {__all__ 额外导出名: 算子名}，仅来自目录扫描（entry-point 形式无 __init__ 可静态解析）
_export_index = dict(_dir_exports)

_ep_overridden_ops = frozenset(_ep_ops)

# {新算子相对路径: [legacy旧名列表]}，算子加载成功后顺手注册 legacy 别名
_legacy_aliases = {}
for _legacy_name, _legacy_target in _legacy_map.items():
    _legacy_aliases.setdefault(_legacy_target, []).append(_legacy_name)


def _resolve_op(name):
    """加载单个算子模块并解析其全部导出属性（与旧版急切加载语义一致）。

    返回 {导出名: 对象}；导入失败时抛出原始异常，由调用方决定降级行为。
    """
    target = _op_registry[name]
    exports = {}
    if ":" in target:
        module_path, attr_name = target.split(":", 1)
        _mod = importlib.import_module(module_path)
        if hasattr(_mod, attr_name):
            exports[name] = getattr(_mod, attr_name)
        elif hasattr(_mod, name):
            exports[name] = getattr(_mod, name)
        else:
            exports[name] = _mod
        return exports
    _mod = importlib.import_module(target)
    if hasattr(_mod, name):
        exports[name] = getattr(_mod, name)
    else:
        exports[name] = _mod
    for _extra in getattr(_mod, "__all__", ()):
        if _extra != name and not _extra.startswith("_"):
            try:
                exports[_extra] = getattr(_mod, _extra)
            except AttributeError:
                pass
    try:
        _gmod = importlib.import_module("%s.graph_convert_%s" % (target, name))
        _func = "convert_%s" % name
        if hasattr(_gmod, _func):
            exports[_func] = getattr(_gmod, _func)
    except ImportError:
        pass
    return exports


def _load_op_attributes(name):
    """按需加载算子并缓存其全部导出属性；失败时 warning + AttributeError（与旧行为一致）。"""
    try:
        exports = _resolve_op(name)
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning("Failed to load op '%s': %s", name, e)
        raise AttributeError(
            "module %r has no attribute %r (op load failed, see warnings above)"
            % (__name__, name)
        ) from e
    globals().update(exports)
    _register_legacy_aliases(name)
    return exports


def _register_legacy_aliases(name):
    """算子加载成功后，把指向该算子的 legacy 旧名注册为 sys.modules 别名。

    与旧版急切加载"导入即注册全部别名"的行为对齐；此处按需注册。
    """
    target = _op_registry.get(name, "")
    if ":" in target or not target.startswith(__name__ + "."):
        return
    rel_target = target[len(__name__) + 1 :]
    real_mod = sys.modules.get(target)
    if real_mod is None:
        return
    for legacy_name in _legacy_aliases.get(rel_target, ()):
        if legacy_name in _ep_overridden_ops:
            continue
        alias = "%s.%s" % (__name__, legacy_name)
        sys.modules.setdefault(alias, real_mod)


def load_all():
    """显式加载全部算子（等价于旧版导入即全量加载的行为）。

    供需要 ``torch.ops.cann_ops_transformer`` 全量 schema 注册、或希望一次性
    完成全部 JIT 编译的场景使用。返回成功加载的算子名列表。
    """
    loaded = []
    for _op_name in sorted(_op_registry):
        try:
            _load_op_attributes(_op_name)
        except AttributeError:
            continue
        loaded.append(_op_name)
    return loaded


def __getattr__(name):
    if name in _op_registry:
        exports = _load_op_attributes(name)
        if name in exports:
            return exports[name]
    elif name in _export_index:
        op_name = _export_index[name]
        exports = _load_op_attributes(op_name)
        if name in exports:
            return exports[name]
    elif name in _convert_funcs:
        op_name = _convert_funcs[name]
        target = _op_registry[op_name]
        if ":" not in target:
            try:
                _gmod = importlib.import_module(
                    "%s.graph_convert_%s" % (target, op_name)
                )
            except ImportError as e:
                logger.warning("Failed to load op '%s': %s", op_name, e)
                raise AttributeError(
                    "module %r has no attribute %r (graph converter load failed)"
                    % (__name__, name)
                ) from e
            if hasattr(_gmod, name):
                globals()[name] = getattr(_gmod, name)
                return globals()[name]
    elif name in _legacy_map:
        if name in _ep_overridden_ops:
            logger.warning(
                "legacy module '%s' is overridden by a package entry point, "
                "skip registering it",
                name,
            )
        else:
            try:
                _new_mod = importlib.import_module(
                    "%s.%s" % (__name__, _legacy_map[name])
                )
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning("Failed to register legacy module '%s': %s", name, e)
                raise AttributeError(
                    "module %r has no attribute %r (legacy op load failed, "
                    "see warnings above)" % (__name__, name)
                ) from e
            sys.modules["%s.%s" % (__name__, name)] = _new_mod
            value = getattr(_new_mod, name, _new_mod)
            globals()[name] = value
            return value
    elif name in _category_names:
        try:
            module = importlib.import_module("%s.%s" % (__name__, name))
        except ImportError as e:
            raise AttributeError(
                "module %r has no attribute %r" % (__name__, name)
            ) from e
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # 排除模块级基础设施名（os/importlib/logging/ast/sys 等）
    _infra = {"os", "importlib", "logging", "ast", "sys"}
    names = {n for n in globals() if not n.startswith("_") and n not in _infra}
    names.update(_op_registry)
    names.update(_export_index)
    names.update(_legacy_map)
    names.update(_category_names)
    names.update(_convert_funcs)
    return sorted(names)


class _LegacyOpAliasLoader:
    """把 legacy 旧名模块加载重定向到真实的新算子模块。"""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return None  # 使用默认模块创建

    def exec_module(self, module):
        real = importlib.import_module(self._real_name)
        # import 语句最终从 sys.modules 取模块；替换为真实模块以完成别名注册
        sys.modules[module.__spec__.name] = real


class _LegacyOpAliasFinder:
    """支持 ``import cann_ops_transformer.ops.<legacy旧名>`` 的懒别名。

    仅精确匹配 ``<本包>.<legacy_map 中的旧名>``（单段名）；其余一律返回
    None 交回默认导入机制。任何内部异常同样返回 None（不劣于默认行为）。
    """

    def find_spec(self, fullname, path=None, target=None):
        try:
            prefix = __name__ + "."
            if not fullname.startswith(prefix):
                return None
            short = fullname[len(prefix) :]
            if "." in short or short not in _legacy_map:
                return None
            if short in _ep_overridden_ops:
                return None
            from importlib.machinery import ModuleSpec

            return ModuleSpec(
                fullname, _LegacyOpAliasLoader(prefix + _legacy_map[short])
            )
        except Exception:
            return None


def _install_legacy_alias_finder():
    if not any(isinstance(finder, _LegacyOpAliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_LegacyOpAliasFinder())


_install_legacy_alias_finder()
