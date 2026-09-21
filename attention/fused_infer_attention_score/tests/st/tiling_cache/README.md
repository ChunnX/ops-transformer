# FIA 手写 ACLNN 的 tiling 缓存（A2/A3）

只修改 ops-transformer_fix，不需要修改或重新编译 opbase。

V1～V4 数组接口保留原有 L2_DFX_PHASE_1。手写 Inner 构建 launcher 后，在
DoOpTilingFusedInferAttentionScore 的 DAV_2201 路由查询 FIA 自有缓存：
未命中执行原 tiling，成功后立即保存；命中直接恢复结果并返回。
第二次 GetWorkspaceSize 可以在第一次 Run 之前复用，不依赖 executor 缓存完成。
opbase 内部 TilingResCache 仍负责单个 launcher 的 GetWorkspaceSize→Run 结果传递；
本实现没有调用该内部类，也没有保存 executor 或设备地址。

缓存键包含属性（softmaxLse 按 bool 读取）、输入/输出 shape 与 dtype/format、
动态 K/V 全列表、stride、相关 view 标记、值依赖输入的完整 host 数据，以及
SoC、核数、UB/L1/L0/L2、系统 workspace 和 deterministic 信息。
可选输入使用逻辑 IR 索引。无法读取完整 host 数据、超过大小限制时绕过缓存，不截断键。

缓存结果包含 tiling blob、tilingKey、blockDim、workspace 大小，以及本次实际设置的
schedule mode。FIA 公共基类、split-fuse、旧 IFA/PFA 的调度模式写入均经过 recorder。
128 条 LRU，键/结果的计费预算 16 MiB（不含容器额外开销和在途读者持有的快照），
单键/单 blob 上限 1 MiB。互斥锁保护容器，不可变快照支持锁外恢复；
失败 tiling 不入缓存。FIA_TILING_CACHE_DISABLE=1 在进程启动时关闭本缓存。

新增/修改 tiling 依赖或输出状态时，必须同步维护 BuildKey/Snapshot/Restore。
DAV_3510、DAV_5102 和 310P 不走此缓存。

## Host UT

新增 test_fused_infer_attention_score_tiling_cache.cpp，已有 arch22 CMake 自动收集：
覆盖 host 值、bool、平台核数、动态列表尾部、超限绕过、跨 context 结果深拷贝与
schedule mode 恢复、LRU 淘汰、并发读写。

在匹配本仓库的 CANN Linux 环境，从仓库根目录执行：

```bash
bash build.sh --ophost_test --ops=fused_infer_attention_score --soc=ascend910b
```

## A2/A3 设备回归

安装本次构建的算子包（包括 tiling 库和 opapi），分别在 A2、A3 执行。
cache_smoke.cpp 只调用公开 V1～V4 数组接口。每组先调用两次 GetWorkspaceSize，
再分别 Run；关闭 opbase executor 缓存并检查返回的 executor 不是缓存包装器。
第二次调用更换设备地址和 V 数据，以数值结果检查是否误用了旧地址。

从本目录编译：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OPAPI_LIB_DIR=/path/to/fixed/lib64
export LD_LIBRARY_PATH="${OPAPI_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"
g++ -std=c++17 -Wno-deprecated-declarations cache_smoke.cpp \
    -I../../../op_api -I"${ASCEND_HOME_PATH}/include" \
    -L"${OPAPI_LIB_DIR}" -L"${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${OPAPI_LIB_DIR}" \
    -lopapi_transformer -lnnopbase -lascendcl -o cache_smoke
ldd ./cache_smoke

set -o pipefail
for version in 1 2 3 4; do
    DISABLE_L2_CACHE=1 FIA_TILING_CACHE_DISABLE=0 \
    ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
        ./cache_smoke "$version" 2>&1 | tee "cache_on_v${version}.log" || exit 1
    DISABLE_L2_CACHE=1 FIA_TILING_CACHE_DISABLE=1 \
    ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
        ./cache_smoke "$version" 2>&1 | tee "cache_off_v${version}.log" || exit 1
done
```

若安装包未提供 opdev 头文件，用匹配版本 opbase 的 include/nnopbase 补充 include 路径。
确认 tiling 库确实来自修改后的算子包；ldd 仅确认链接库，不能证明 tiling 加载路径。

检查每个进程三组 BEGIN pair 与 BOTH GWS DONE BEFORE RUN 之间的日志：

| 组 | 参数 | 第一次 GWS | 第二次 GWS |
| --- | --- | --- | --- |
| 1 | KV length=4, scale=1 | FIA tiling cache store | FIA tiling cache hit |
| 2 | KV length=2, scale=1 | FIA tiling cache store | FIA tiling cache hit |
| 3 | KV length=4, scale=0.5 | FIA tiling cache store | FIA tiling cache hit |

关闭 FIA 缓存时不应出现上述 store/hit。两种模式都必须输出三次 PASS pair。
单有数值 PASS 不足以证明缓存命中，还必须核对日志。组 2 检查 shape 不变但 host
序列长度变化必须 miss；组 3 检查属性变化必须 miss。

当前 Windows 工作区没有匹配的 CANN 编译/设备环境，尚未实际执行这些 UT/ST。
上线前还需设备验证非连续 PA/MLA、softmaxLse=true、多流和变核数等业务场景。
