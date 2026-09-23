# KvCacheTurboQuant · 挂仓编译与 NPU smoke（可复制）

> 写码真源：`/root/ascend-work/deliver/kv_cache_turbo_quant/`  
> 仓内入口：软链 → `refs/ops-transformer/experimental/attention/kv_cache_turbo_quant`

## 直白版（你要干什么）

1. **环境**：`source` CANN  
2. **挂目录**：软链 deliver 到 ops-transformer（已做过可跳过）  
3. **编译**：`build.sh --pkg ...` 出 `.run`  
4. **安装**：跑 `.run` + `source set_env.bash`  
5. **上板 smoke**：编并跑 `examples/test_aclnn_kv_cache_turbo_quant.cpp`

本机 A3 用 `--soc=ascend910_93`；交验收 A2 机改 `--soc=ascend910b`。

---

## 已在本机跑通的命令（2026-09-23）

```bash
# 0) Python：build 需要带 pip 的 python3（本机用 3.12 顶到 PATH）
export PATH=/root/ascend-work/.localbin:$PATH   # python3 → 3.12.13
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 软链
ln -sfn /root/ascend-work/deliver/kv_cache_turbo_quant \
  /root/ascend-work/refs/ops-transformer/experimental/attention/kv_cache_turbo_quant

# 2) 编译自定义包
cd /root/ascend-work/refs/ops-transformer
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=kv_cache_turbo_quant -j8
# 产物：build_out/cann-ops-transformer-custom_linux-aarch64.run

# 3) 安装
cd build_out
bash ./cann-ops-transformer-custom_linux-aarch64.run --quiet \
  --install-path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/vendors/custom_transformer/bin/set_env.bash

# 4) 孪生期望
cd /root/ascend-work/deliver/kv_cache_turbo_quant/tests
python3 gen_selftest_matrices.py
g++ -O2 -std=c++17 -o cpu_encode_twin cpu_encode_twin.cpp && ./cpu_encode_twin

# 5) NPU smoke
g++ -O2 -std=c++17 \
  ../examples/test_aclnn_kv_cache_turbo_quant.cpp \
  -I/usr/local/Ascend/ascend-toolkit/latest/vendors/custom_transformer/op_api/include \
  -I${ASCEND_HOME_PATH}/include \
  -L/usr/local/Ascend/ascend-toolkit/latest/vendors/custom_transformer/op_api/lib \
  -L${ASCEND_HOME_PATH}/lib64 \
  -lcust_opapi -lascendcl -lnnopbase \
  -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/vendors/custom_transformer/op_api/lib:${ASCEND_HOME_PATH}/lib64 \
  -o /tmp/test_aclnn_kv_cache_turbo_quant

/tmp/test_aclnn_kv_cache_turbo_quant \
  /root/ascend-work/deliver/kv_cache_turbo_quant/tests/data
```

**本机证据摘要：**

```text
idx0=157 qjl0=230
PASS npu idx/qjl == twin_smoke_out (exact)
ALL PASS test_aclnn_kv_cache_turbo_quant smoke
```

---

## 踩过的坑（以后别再踩）

| 坑 | 处理 |
|----|------|
| `refs/ops-transformer` 曾 sparse 只有 attention | `git sparse-checkout disable` |
| 系统 python3.10 无 pip / packaging | PATH 优先用本机 python3.12 |
| Kernel 缺 `using namespace AscendC` | 已补 |
| `static_cast` bf16 ↔ float | 改 `ToFloat` + `Cast(...CAST_RINT)` |
| `build.sh --run_example` 找不到 example | 手动按上面 g++ 链接 `-lcust_opapi` |

## 你还需要做什么？

**功能冒烟：不用再做。**  
下一阶段才是：精度全 case、性能 ≥10×、设计文档 / 腾讯报告 / 个人仓邀审。
