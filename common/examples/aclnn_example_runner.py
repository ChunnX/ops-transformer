# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if home:
        return Path(home)
    return Path("/home/developer/Ascend/cann-9.0.0")


def _find_custom_opp() -> Path:
    candidates = []
    for env_name in ("ASCEND_CUSTOM_OPP_PATH",):
        env = os.environ.get(env_name, "")
        candidates.extend([Path(p) for p in env.split(":") if p])
    opp_path = os.environ.get("ASCEND_OPP_PATH")
    if opp_path:
        cfg = Path(opp_path) / "vendors" / "config.ini"
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith("load_priority="):
                    for vendor in line.split("=", 1)[1].split(","):
                        vendor = vendor.strip()
                        if vendor:
                            candidates.append(Path(opp_path) / "vendors" / vendor)
    candidates.extend([
        Path("/home/developer/Ascend/cann-9.0.0/opp/vendors/migrated_transformer"),
        Path("/tmp/ops_transformer_opp_migrated/vendors/migrated_transformer"),
    ])
    for path in candidates:
        if (path / "op_api/lib/libcust_opapi.so").exists():
            return path
    raise RuntimeError("cannot find custom op_api libcust_opapi.so")


def acl_dtype(tensor: torch.Tensor) -> int:
    dtype = tensor.dtype
    if dtype is torch.float32:
        return 0
    if dtype is torch.float16:
        return 1
    if dtype is torch.int32:
        return 3
    if dtype is torch.int64:
        return 9
    if dtype is torch.bfloat16:
        return 27
    if dtype is torch.float8_e5m2:
        return 35
    if dtype is torch.float8_e4m3fn:
        return 36
    if dtype is torch.float8_e8m0fnu:
        return 37
    if str(dtype) in ("torch_npu.hifloat8", "hifloat8") or getattr(tensor, "dtype", None) == 290:
        return 34
    # torch_npu.hifloat8 is represented as a private dtype in some builds; callers can pass dtype code explicitly
    raise TypeError(f"unsupported dtype for aclnn example runner: {dtype}")


def _write_cpp(build_dir: Path) -> Path:
    source = build_dir / "aclnn_example_runner.cpp"
    source.write_text(r'''
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include <acl/acl.h>
#include <aclnnop/aclnn_hc_pre.h>
#include <aclnnop/aclnn_hc_post.h>
#include <aclnnop/aclnn_moe_gating_top_k_hash.h>
#include <aclnnop/aclnn_kv_compress_epilog.h>
#include <aclnnop/aclnn_indexer_compress_epilog.h>

namespace {

void CheckAcl(aclnnStatus ret, const char *what) {
  if (ret != ACL_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed, ret=" + std::to_string(ret));
  }
}

std::vector<int64_t> Shape(const at::Tensor &t) {
  return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

std::vector<int64_t> Strides(const at::Tensor &t) {
  return std::vector<int64_t>(t.strides().begin(), t.strides().end());
}

aclTensor *CreateAclTensor(const at::Tensor &t, int64_t dtype) {
  auto shape = Shape(t);
  auto strides = Strides(t);
  return aclCreateTensor(shape.data(), shape.size(), static_cast<aclDataType>(dtype),
                         strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                         const_cast<void *>(t.data_ptr()));
}

struct TensorHolder {
  aclTensor *tensor = nullptr;
  TensorHolder() = default;
  TensorHolder(const at::Tensor &t, int64_t dtype) : tensor(CreateAclTensor(t, dtype)) {}
  TensorHolder(const TensorHolder &) = delete;
  TensorHolder &operator=(const TensorHolder &) = delete;
  TensorHolder(TensorHolder &&other) noexcept : tensor(other.tensor) {
    other.tensor = nullptr;
  }
  TensorHolder &operator=(TensorHolder &&other) noexcept {
    if (this != &other) {
      if (tensor != nullptr) {
        aclDestroyTensor(tensor);
      }
      tensor = other.tensor;
      other.tensor = nullptr;
    }
    return *this;
  }
  ~TensorHolder() {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
    }
  }
  operator aclTensor *() const { return tensor; }
  operator const aclTensor *() const { return tensor; }
};

void RunExecutor(void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
                 aclrtStream stream, aclnnStatus (*fn)(void *, uint64_t, aclOpExecutor *, aclrtStream),
                 const char *name) {
  CheckAcl(fn(workspace, workspace_size, executor, stream), name);
  CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

void *MallocWorkspace(uint64_t workspace_size) {
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtMalloc workspace failed, ret=" + std::to_string(ret));
    }
  }
  return workspace;
}

}  // namespace

void hc_pre(const at::Tensor &x, int64_t x_dtype,
            const at::Tensor &hc_fn, int64_t hc_fn_dtype,
            const at::Tensor &hc_scale, int64_t hc_scale_dtype,
            const at::Tensor &hc_base, int64_t hc_base_dtype,
            int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps, double norm_eps,
            const at::Tensor &y_out, int64_t y_dtype,
            const at::Tensor &post_out, int64_t post_dtype,
            const at::Tensor &comb_frag_out, int64_t comb_dtype) {
  TensorHolder x_acl(x, x_dtype), hc_fn_acl(hc_fn, hc_fn_dtype), hc_scale_acl(hc_scale, hc_scale_dtype);
  TensorHolder hc_base_acl(hc_base, hc_base_dtype), y_acl(y_out, y_dtype), post_acl(post_out, post_dtype);
  TensorHolder comb_acl(comb_frag_out, comb_dtype);
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  CheckAcl(aclnnHcPreGetWorkspaceSize(x_acl, hc_fn_acl, hc_scale_acl, hc_base_acl, hc_mult,
                                      hc_sinkhorn_iters, hc_eps, norm_eps, y_acl, post_acl, comb_acl,
                                      &workspace_size, &executor),
           "aclnnHcPreGetWorkspaceSize");
  void *workspace = MallocWorkspace(workspace_size);
  aclrtStream stream = nullptr;
  CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  RunExecutor(workspace, workspace_size, executor, stream, aclnnHcPre, "aclnnHcPre");
  aclrtDestroyStream(stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

void hc_post(const at::Tensor &x, int64_t x_dtype,
             const at::Tensor &residual, int64_t residual_dtype,
             const at::Tensor &post, int64_t post_dtype,
             const at::Tensor &comb, int64_t comb_dtype,
             const at::Tensor &out, int64_t out_dtype) {
  TensorHolder x_acl(x, x_dtype), residual_acl(residual, residual_dtype), post_acl(post, post_dtype);
  TensorHolder comb_acl(comb, comb_dtype), out_acl(out, out_dtype);
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  CheckAcl(aclnnHcPostGetWorkspaceSize(x_acl, residual_acl, post_acl, comb_acl, out_acl,
                                       &workspace_size, &executor),
           "aclnnHcPostGetWorkspaceSize");
  void *workspace = MallocWorkspace(workspace_size);
  aclrtStream stream = nullptr;
  CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  RunExecutor(workspace, workspace_size, executor, stream, aclnnHcPost, "aclnnHcPost");
  aclrtDestroyStream(stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

void moe_gating_top_k_hash(const at::Tensor &x, int64_t x_dtype,
                           const c10::optional<at::Tensor> &bias, int64_t bias_dtype,
                           const c10::optional<at::Tensor> &input_ids, int64_t input_ids_dtype,
                           const c10::optional<at::Tensor> &tid2eid, int64_t tid2eid_dtype,
                           int64_t k, int64_t k_group, int64_t group_count,
                           int64_t group_select_mode, int64_t renorm, int64_t norm_type,
                           bool out_flag, double routed_scaling_factor, double eps,
                           const at::Tensor &y_out, int64_t y_dtype,
                           const at::Tensor &expert_idx_out, int64_t expert_idx_dtype,
                           const at::Tensor &out_out, int64_t out_dtype) {
  TensorHolder x_acl(x, x_dtype), y_acl(y_out, y_dtype), idx_acl(expert_idx_out, expert_idx_dtype);
  TensorHolder out_acl(out_out, out_dtype);
  TensorHolder bias_acl, input_acl, tid_acl;
  if (bias.has_value()) {
    bias_acl = TensorHolder(bias.value(), bias_dtype);
  }
  if (input_ids.has_value()) {
    input_acl = TensorHolder(input_ids.value(), input_ids_dtype);
  }
  if (tid2eid.has_value()) {
    tid_acl = TensorHolder(tid2eid.value(), tid2eid_dtype);
  }
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  CheckAcl(aclnnMoeGatingTopKHashGetWorkspaceSize(x_acl, bias_acl.tensor, input_acl.tensor, tid_acl.tensor,
                                                  k, k_group, group_count, group_select_mode, renorm,
                                                  norm_type, out_flag, routed_scaling_factor, eps,
                                                  y_acl, idx_acl, out_acl, &workspace_size, &executor),
           "aclnnMoeGatingTopKHashGetWorkspaceSize");
  void *workspace = MallocWorkspace(workspace_size);
  aclrtStream stream = nullptr;
  CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  RunExecutor(workspace, workspace_size, executor, stream, aclnnMoeGatingTopKHash, "aclnnMoeGatingTopKHash");
  aclrtDestroyStream(stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

void kv_compress_epilog(const at::Tensor &kv_cache, int64_t kv_dtype,
                        const at::Tensor &x, int64_t x_dtype,
                        const at::Tensor &slot_mapping, int64_t slot_dtype,
                        int64_t quant_group_size, int64_t quant_mode, int64_t round_scale,
                        double scale) {
  TensorHolder kv_acl(kv_cache, kv_dtype), x_acl(x, x_dtype), slot_acl(slot_mapping, slot_dtype);
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  CheckAcl(aclnnKvCompressEpilogGetWorkspaceSize(kv_acl, x_acl, slot_acl, quant_group_size, quant_mode,
                                                 round_scale, scale, &workspace_size, &executor),
           "aclnnKvCompressEpilogGetWorkspaceSize");
  void *workspace = MallocWorkspace(workspace_size);
  aclrtStream stream = nullptr;
  CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  RunExecutor(workspace, workspace_size, executor, stream, aclnnKvCompressEpilog, "aclnnKvCompressEpilog");
  aclrtDestroyStream(stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

void indexer_compress_epilog(const at::Tensor &cache, int64_t cache_dtype,
                             const at::Tensor &cache_scale, int64_t cache_scale_dtype,
                             const at::Tensor &x, int64_t x_dtype,
                             const at::Tensor &slot_mapping, int64_t slot_dtype,
                             int64_t quant_mode, bool round_scale, double scale) {
  TensorHolder cache_acl(cache, cache_dtype), scale_acl(cache_scale, cache_scale_dtype);
  TensorHolder x_acl(x, x_dtype), slot_acl(slot_mapping, slot_dtype);
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  CheckAcl(aclnnIndexerCompressEpilogGetWorkspaceSize(cache_acl, scale_acl, x_acl, slot_acl, quant_mode,
                                                      round_scale, scale, &workspace_size, &executor),
           "aclnnIndexerCompressEpilogGetWorkspaceSize");
  void *workspace = MallocWorkspace(workspace_size);
  aclrtStream stream = nullptr;
  CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  RunExecutor(workspace, workspace_size, executor, stream, aclnnIndexerCompressEpilog,
              "aclnnIndexerCompressEpilog");
  aclrtDestroyStream(stream);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hc_pre", &hc_pre);
  m.def("hc_post", &hc_post);
  m.def("moe_gating_top_k_hash", &moe_gating_top_k_hash);
  m.def("kv_compress_epilog", &kv_compress_epilog);
  m.def("indexer_compress_epilog", &indexer_compress_epilog);
}
''')
    return source


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    ascend = _ascend_home()
    custom_opp = _find_custom_opp()
    build_dir = _repo_root() / "build" / "aclnn_example_runner"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = _write_cpp(build_dir)
    include_dirs = [
        str(ascend / "include"),
        str(ascend / "x86_64-linux/include"),
        str(custom_opp / "op_api/include"),
    ]
    library_dirs = [
        str(ascend / "lib64"),
        str(ascend / "x86_64-linux/lib64"),
        str(custom_opp / "op_api/lib"),
    ]
    _EXT = load(
        name="aclnn_example_runner",
        sources=[str(source)],
        extra_include_paths=include_dirs,
        extra_ldflags=[
            *[f"-L{path}" for path in library_dirs],
            "-lcust_opapi",
            "-lascendcl",
            f"-Wl,-rpath,{custom_opp / 'op_api/lib'}",
            f"-Wl,-rpath,{ascend / 'lib64'}",
            f"-Wl,-rpath,{ascend / 'x86_64-linux/lib64'}",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    return _EXT


def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps):
    ext = _load_ext()
    y_shape = (*x.shape[:-2], x.shape[-1]) if x.dim() == 4 else (x.shape[0], x.shape[-1])
    post_shape = (*x.shape[:-2], hc_mult) if x.dim() == 4 else (x.shape[0], hc_mult)
    comb_shape = (*x.shape[:-2], hc_mult, hc_mult) if x.dim() == 4 else (x.shape[0], hc_mult, hc_mult)
    y = torch.empty(y_shape, dtype=x.dtype, device=x.device)
    post = torch.empty(post_shape, dtype=torch.float32, device=x.device)
    comb = torch.empty(comb_shape, dtype=torch.float32, device=x.device)
    ext.hc_pre(x.contiguous(), acl_dtype(x), hc_fn.contiguous(), acl_dtype(hc_fn),
               hc_scale.contiguous(), acl_dtype(hc_scale), hc_base.contiguous(), acl_dtype(hc_base),
               int(hc_mult), int(hc_sinkhorn_iters), float(hc_eps), float(norm_eps),
               y, acl_dtype(y), post, acl_dtype(post), comb, acl_dtype(comb))
    return y, post, comb


def hc_post(x, residual, post, comb):
    ext = _load_ext()
    out = torch.empty_like(residual)
    ext.hc_post(x.contiguous(), acl_dtype(x), residual.contiguous(), acl_dtype(residual),
                post.contiguous(), acl_dtype(post), comb.contiguous(), acl_dtype(comb),
                out, acl_dtype(out))
    return out


def moe_gating_top_k_hash(x, k, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1,
                          routed_scaling_factor=1.0, eps=1e-6, group_select_mode=0, renorm=0,
                          norm_type=1, out_flag=False):
    ext = _load_ext()
    y = torch.empty((*x.shape[:-1], k), dtype=x.dtype, device=x.device)
    idx = torch.empty((*x.shape[:-1], k), dtype=torch.int32, device=x.device)
    out = torch.empty_like(x, dtype=torch.float32)
    ext.moe_gating_top_k_hash(
        x.contiguous(), acl_dtype(x),
        bias.contiguous() if bias is not None else None, acl_dtype(bias) if bias is not None else 0,
        input_ids.contiguous() if input_ids is not None else None, acl_dtype(input_ids) if input_ids is not None else 0,
        tid2eid.contiguous() if tid2eid is not None else None, acl_dtype(tid2eid) if tid2eid is not None else 0,
        int(k), int(k_group), int(group_count), int(group_select_mode), int(renorm), int(norm_type),
        bool(out_flag), float(routed_scaling_factor), float(eps), y, acl_dtype(y), idx, acl_dtype(idx),
        out, acl_dtype(out))
    return y, idx, out


def kv_compress_epilog(kv_cache, x, slot_mapping, quant_group_size=128, quant_mode=1,
                       round_scale_flag=True, scale=1.0, kv_cache_dtype=None):
    ext = _load_ext()
    dtype = int(kv_cache_dtype) if kv_cache_dtype is not None else acl_dtype(kv_cache)
    ext.kv_compress_epilog(kv_cache, dtype, x.contiguous(), acl_dtype(x),
                           slot_mapping.contiguous(), acl_dtype(slot_mapping), int(quant_group_size),
                           int(quant_mode), int(round_scale_flag), float(scale))
    return kv_cache


def indexer_compress_epilog(cache, cache_scale, x, slot_mapping, quant_mode=1, round_scale=True,
                            scale=1.0, cache_dtype=None, cache_scale_dtype=None):
    ext = _load_ext()
    ext.indexer_compress_epilog(
        cache, int(cache_dtype) if cache_dtype is not None else acl_dtype(cache),
        cache_scale, int(cache_scale_dtype) if cache_scale_dtype is not None else acl_dtype(cache_scale),
        x.contiguous(), acl_dtype(x), slot_mapping.contiguous(), acl_dtype(slot_mapping),
        int(quant_mode), bool(round_scale), float(scale))
    return cache, cache_scale
