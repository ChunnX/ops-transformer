# SageAttention2 Host Tiling 与接口编码协议

本目录承载 SageAttention2 的闭式 Host Tiling 计算核心与接口编码协议（交付态两段式
ACLNN C 接口唯一实现在 `../op_api/aclnn_sage_attention2.cpp`，字符串属性 ABI，
随 run 包安装——本目录不再保留任何第二套两段式实现）。

## 1. 文件清单

| 文件 | 职责 |
|---|---|
| `sage_attn_host_tiling.h` | 闭式微秒级 Host Tiling 计算核心 `ComputeHostTiling`（inline 唯一定义）+ 双 Die 亲和调度判定 + aclnn 状态码词汇表（CANN 头缺失时的对齐桩） |
| `sage_attn_proto.h` | "字符串属性 → 数值编码"唯一映射（`*CodeFromString`）与冻结签名默认值（`kDefault*`）；数值编码枚举唯一定义在 `../sageattn_fp8/ascendc/tiling.h` |
| `autotune_tiling_bank.h` | P-01..P-07 经验调优知识库（`QueryTunedTiling`），未命中平滑回退闭式推导 |

## 2. 单一实现纪律

- `ComputeHostTiling` 被 `../op_host/sage_attention2_tiling.cpp`（gert TilingImpl）
  与宿主单测 `../tests/test_aclnn_host.cpp` 共同包含，保证 TilingImpl 下发位元与
  单测锁值一致；op_api 层经框架 TilingImpl 取同一份 tiling。
- 参数校验唯一点：闭式计算核心内的域校验（head_dim、GQA 整除、gran/accum/layout
  枚举域、causal 要求 Sq==Sk、sm_scale 非正值拒绝）+ 资源台账 fail-fast
  （`InspectTilingPlan`）。Python 边界（`error_check`）与 op_api 白名单按同一域拒绝，
  不在第三处复制。

## 3. 两段式 C 接口（交付态签名）

第一段 `aclnnSageAttention2GetWorkspaceSize` 与第二段 `aclnnSageAttention2` 的
唯一权威签名见 `../op_api/aclnn_sage_attention2.h`：

- 属性为字符串形态：`tensorLayout`（`"HND"`/`"NHD"`，非法取值报错）、
  `qkQuantGran`（`"per_thread"`/`"per_warp"`）、`pvAccumDtype`
  （`"fp32"`/`"fp32+fp32"`/`"fp32+fp16"`），顺序与 `op_host/sage_attention2_def.cpp`
  注册的 7 项属性一致；
- `smScale` 冻结语义：默认 `1/sqrt(head_dim)` 仅由上层桥接在 `sm_scale=None` 时
  折叠后传入，C 入口收到非正值/非有限值返回 `ACLNN_ERR_PARAM_INVALID`；
- 数值编码（0=NHD/1=HND、2=per_warp/3=per_thread、0/1/2 累加档）仅供 Host 内部
  消费（TilingImpl/设备核），经本目录 `sage_attn_proto.h` 的 `*CodeFromString`
  唯一映射产生，禁止作为跨语言 ABI 形参再引入。

调用示例（真机两段式）见 `../benchmarks/benchmark_aclnn.cpp`（dlsym 交付态符号，
函数指针签名逐参数对齐 `op_api/aclnn_sage_attention2.h`）。

## 4. 验证

- `../tests/test_aclnn_host.cpp`：直测 `ComputeHostTiling`/`TilingData`——结构体
  160 字节/32 字节对齐与逐字段偏移锁值、标准 GQA 4:1 分核与 Workspace 512B 对齐、
  fp32+smooth_v 空间入账、Split-KV 触发、28 核减配拓扑、非法入参阻断（GQA/头维/
  枚举域/layout/causal-sm_scale）、双 Die 亲和调度与调优库命中/回退。
- TilingData 结构契约另有 Python 镜像锁值：`../tests/test_tiling_struct.py`
  （ctypes 逐字段偏移）与 `../tests/tiling_mirror.py`（唯一定义点）。
