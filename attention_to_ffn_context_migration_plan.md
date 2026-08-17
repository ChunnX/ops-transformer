# AttentionToFFN A5 显式 Context 迁移计划（重审版）

> 本文中的 **A5** 指 Ascend950 / arch35，**V3** 指新增的显式 Context 算子接口。两者不是同一个概念：当前 `AttentionToFFN` 已有 A5 kernel，但仍是 `group + 隐式 HCCL Context` 路径，并不是本文要做的 V3。

## 1. 结论

推荐采用一套“组合方案”，而不是完整照抄某一个算子：

| 层次 | 借鉴对象 | AttentionToFFN 的做法 |
|---|---|---|
| V2/V3 接口拆分 | Dispatch/Combine | 保留现有 `AttentionToFFN`，新增独立 `AttentionToFFNV3` |
| Host 路由与缓存 | Dispatch/Combine | `group` 只在兼容层使用；专用 Builder 缓存 Engine/device Context，并在每次调用时包装 Tensor |
| Kernel 解析 | Engram、Combine A5 | 不定义 ContextHolder，首个输入直接强转为 Context POD 指针 |
| 通信内存 | AttentionToFFN 当前实现 | 继续使用原始 HCCL/UserMem 窗口，不照搬 Engram 的 Host pinned 外挂内存 |
| 业务布局 | AttentionToFFN 当前实现 | TokenInfo/TokenData 的 offset 不变，仍是 `windowBase + 原 offset` |

最终形成两条互不干扰的路径：

```text
旧路径（保持不变）
AttentionToFFN(group, ...)
  → MC2/KFC 注入 HcclOpResParam
  → 旧 kernel 调 GetHcclContext()

新路径（只注册 A5）
AttentionToFFNV3(context, ...)
  → context 是普通 INT32/ND Tensor
  → V3 kernel 直接 cast 为 AttentionFfnContext *
  → windowAddr[rank] + 原有业务 offset
```

不建议把现有 OpDef 原地改成 V3。一个算子 schema 不能只在 A5 上多出一个 `context` 输入；参照 Dispatch/Combine，应新增独立 V3 算子目录，旧算子继续承担兼容和回归。

## 2. 为什么这样选

### 2.1 Engram 提供“无 ContextHolder”的直接模式

Engram 把 Context 注册成普通必选 Tensor：

- [Engram OpDef 的 `commContext` 输入](mc2/engram_fetch/op_host/engram_fetch_def.cpp#L19)
- [ACLNN 将 Context 直接传给 inner 算子](mc2/engram_fetch/op_api/aclnn_engram_fetch.cpp#L45)
- [Kernel 入口将 Context 作为第一个 GM_ADDR](mc2/engram_fetch/op_kernel/arch35/engram_fetch.cpp#L28)
- [Kernel 直接 cast 并读取 rank](mc2/engram_fetch/op_kernel/arch35/engram_fetch_arch35.h#L92)

这说明显式 Context 不要求额外定义 ContextHolder。

但 Engram 的地址表指向它自己申请、映射和注册的 Host pinned 内存：

- [申请并映射 Host Buffer](torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp#L231)
- [向 HCCL 注册该 Buffer](torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp#L246)
- [按 rank 填地址表与 handle](torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp#L318)

因此只借鉴 Engram 的“薄 Context Tensor + Kernel 直接解释”模式，不照搬它的内存来源和 `ReadNbi/Commit/Drain` 通信协议。

### 2.2 Dispatch/Combine 提供“V2 兼容转 V3”的 Host 模式

Dispatch/Combine 的旧公开 API 在 A5 MTE 路径上创建 Context，然后调用内部 V3：

- [Dispatch：A5 创建 Context 并调用 inner V3](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L189)
- [Combine：A5 创建 Context 并调用 inner V3](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L186)
- [公共 Engine Context 的创建、缓存查询和 Tensor 封装](mc2/common/op_api/mc2_context.cpp#L499)

V3 schema 的共同变化是首增 `context`、删除通信 group、增加 `ccl_buffer_size`：

- [DispatchV3 OpDef](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L19)
- [DispatchV3 proto](mc2/moe_distribute_dispatch_v3/op_graph/moe_distribute_dispatch_v3_proto.h#L73)
- [DispatchV3 tiling 仅配置新索引并复用公共实现](mc2/moe_distribute_dispatch_v3/op_host/op_tiling/moe_distribute_dispatch_v3_tiling.cpp#L37)

DispatchV3 的确复用了 Holder，但 Holder 定义在 V2 目录：[Mc2MoeContextHolder](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L35)。这只是为了让一套复杂 Dispatch 核心在两种 Context 间切换，不是 V3 的强制要求。Combine A5 已展示直接解释 Context 的另一种写法：[Combine A5 直接 cast `Mc2MoeContext`](mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h#L440)。

## 3. 不能直接复用 Mc2MoeContext

不要把 `GetMc2ContextTensor()` 返回值当成 `AttentionFfnContext` 直接使用，原因有三点。

第一，结构体 ABI 不同。`Mc2MoeContext` 多出 `kfcContextAddr`，第二个字段还是 `rankSizePerServer` 而不是整个通信域的 world size：[Mc2MoeContext 定义](mc2/common/op_kernel/mc2_moe_context.h#L22)。如果按当前自建的 [AttentionFfnContext](mc2/attention_to_ffn/attention_ffn_context.h#L8) 解释，地址数组起点会错位。

第二，地址语义不同。公共 MoE Builder 会给 rank `i` 的 HCCL buffer 基址额外加上 `i * 1024`：[MoE Context 地址填充](mc2/common/op_api/mc2_context.cpp#L281)。MoE kernel 还会把状态区和数据区分开：[Mc2MoeContextHolder 的 `STATE_SIZE` 偏移](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L67)。

AttentionToFFN 当前则直接把 `windowsIn` 或 `userMemRes[i].addr` 当作业务窗口零点：

- [第一次取得远端窗口](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L417)
- [发送结束标记时再次取得远端窗口](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L577)
- [本 rank 直接使用 `localWindowsIn`](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L598)

第三，当前 Attention kernel 没有使用任何 Hcomm handle；而公共 `Mc2Context` 的这条创建路径也没有填 `hcommHandle_[]`。因此不能为了结构看起来像 Engram 而假设 handle 已经可用。

结论：可以复用 `Mc2Context` 的 HCCL 符号加载、Engine Context 缓存、Tensor 封装思路，但必须创建 Attention/FFN 专用 ABI 和专用 Builder。

## 4. Context ABI

### 4.1 建议放置位置

建议把唯一的 Host/Kernel 共用定义放到：

```text
mc2/common/op_kernel/attention_ffn_context.h
```

这个头文件应设计成可供未来 `FFNToAttention` A5/V3 复用，但本阶段只要求 `AttentionToFFN` 使用。不要照搬 Engram 在 Host 和 Kernel 各定义一份结构体的方式；Engram 两份定义甚至使用了不同字段名：

- [Host 的 EngramCommContext](torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp#L113)
- [Kernel 的 EngramCommContext](mc2/engram_fetch/op_kernel/engram_fetch_utils.h#L36)

当前未跟踪文件 [mc2/attention_to_ffn/attention_ffn_context.h](mc2/attention_to_ffn/attention_ffn_context.h#L1) 还存在两个问题：`HCCL_MAX_RANK_SIZE` 没有来源，且 `hcommHandle[]` 当前没有消费者。建议不要直接以它作为最终 ABI。

### 4.2 最小结构

当前 MTE 远端写路径只需要 rank 信息和每个 rank 的原始窗口基址：

```cpp
#include "mc2_moe_context.h" // 复用 HCCL_MAX_RANK_SIZE；也可将该上限再下沉到更通用的公共头

namespace Mc2Aclnn {

struct alignas(8) AttentionFfnContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t windowAddr[HCCL_MAX_RANK_SIZE];
};

static_assert(sizeof(AttentionFfnContext) % sizeof(int32_t) == 0);

} // namespace Mc2Aclnn
```

以当前最大 rank 数 1024 计算，这个最小结构是 8200B，即 2050 个 INT32 元素；实现仍必须从 `sizeof` 推导，不能把 2050 写死在多处。

要求：

- `windowAddr[i]` 必须是旧路径在 rank `i` 上使用的原始业务窗口基址；
- 不加 `i * 1024`；
- 不加 MoE 的 `STATE_SIZE`；
- 不把 TokenInfo/TokenData 本身塞进 Context；
- Tensor 长度必须由 `sizeof(AttentionFfnContext)` 计算，不能硬编码；
- Host、Kernel、Torch 扩展必须包含同一份 ABI 头。

如果后续把通信算法改成 Hcomm `WriteNbi/Commit/Drain`，再为新 ABI 增加 handle；那属于通信算法升级，不是本次最小 Context 切换。

## 5. 通信窗口与外挂内存

本次最小迁移不新增 Engram 式外挂内存，但必须完成“窗口资源显式化”。

```text
Context Tensor
  ├─ rankId / rankSize
  └─ windowAddr[rank]
          │
          └─ 指向已经存在并可被远端访问的 HCCL/UserMem 窗口
                 ├─ TokenInfo 区
                 └─ 512B 对齐后的 TokenData 区
```

现有业务布局继续由这些代码决定：

- [TokenInfo/TokenData 窗口 offset](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L269)
- [Tiling 计算 TokenInfo 与 TokenData 大小](mc2/attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling.cpp#L760)

专用 Builder 需要：

1. 根据 `group` 取得 communicator、rankId、rankSize；
2. 取得每个 rank 的原始 HCCL window 地址；
3. 若支持 UserMem，取得与旧 `userMemRes[i].addr` 相同的地址；
4. 用稳定 tag 创建/查询 Engine Context；
5. 将 POD 拷入 Device Context，并包装成 `INT32/ND` Tensor；
6. 检查所有 local/peer window 的可用容量；第一版建议容量不一致就拒绝；
7. 返回这组 raw window 共同可安全访问的 `ccl_buffer_size`，不减 MoE 状态区；
8. 缓存 HCCL Engine/device Context；每次 GetWorkspaceSize 重新把 `ctx` 包装成 `aclTensor`，由本次 executor 管理 Tensor 描述符的生命周期。

第一版推荐使用 Attention 专属 tag，例如 `group + attention_to_ffn_v3`。只有等 `FFNToAttention` 完成 A5/V3 迁移，并证明双方的 ABI、Engine、窗口协议和生命周期完全一致后，才改成共享 pair tag。

如果现有 HCCL API 无法取得 UserMem 的等价地址，第一版 V3 应明确只支持默认 HCCL window，并在 Host 侧拒绝 UserMem；不能静默退回 `HcclGetHcclBuffer()`。

## 6. 第零步：先证明新旧地址等价  证明不了先这样吧

这是开始大规模改代码前必须完成的小验证。至少在两张卡上比较：

```text
new.windowAddr[localRank] == old.localWindowsIn

userMemType == 0:
new.windowAddr[peer] == old.remoteRes[peer].windowsIn

userMemType != 0:
new.windowAddr[peer] == old.userMemRes[peer].addr
```

同时验证：

- 所有实际访问的 `windowAddr[i]` 非零；
- `rankId < rankSize`；
- `rankSize == world_size`；
- 各 rank 的窗口大小一致；
- `ccl_buffer_size` 是每个 `windowAddr[i]` 共同可访问的完整 raw window 容量。

若上述等价关系不成立，应先修正专用 Builder 或明确 V3 的支持范围，不要通过在 Kernel 中尝试 `+1024`、`+STATE_SIZE` 来猜地址。

这里还要区分两类“远端地址”：旧 `windowsIn/userMemRes.addr` 是当前 AIV `DataCopyPad` 可直接访问的窗口；Engram 通过 channel 查询到的 registered remote memory 则与 handle 一起交给 `ReadNbi/Commit`。如果 Builder 只能得到后一类地址，就不能继续使用当前 DataCopy 算法，必须另立 Hcomm 通信改造任务，本次 Context 迁移应在此处停止。

仅比较地址数值还不够。Phase 0 必须覆盖 **V3-only 冷启动**：进程启动后不先运行旧 `AttentionToFFN`，只由新 Builder 创建资源，然后真实执行跨 rank MTE 写、flag 轮询和读取，并验证权限、可见性、同步顺序及 Context 生命周期。否则可能只是旧 MC2/KFC 路径先创建了资源，导致新路径“碰巧可用”。

## 7. 新增独立 AttentionToFFNV3 算子

建议新增目录：

```text
mc2/attention_to_ffn_v3/
├─ CMakeLists.txt
├─ op_api/
├─ op_graph/
├─ op_host/
├─ op_kernel/arch35/
└─ tests/
```

下文暂用 `AttentionToFFNV3` / `aclnnAttentionToFFNV3` 命名；最终名称以正式接口规格为准。

V3 schema：

```text
新增首个输入：context，INT32，ND，REQUIRED
删除属性：group
保留属性：world_size、三个业务 shape、moe_expert_num、quant_mode、sync_flag、ffn_start_rank_id
新增属性：ccl_buffer_size
只注册 ascend950
不调用 this->MC2().HcclGroup(...)
```

现有 schema 的修改参照位置：

- [当前 OpDef 输入和属性](mc2/attention_to_ffn/op_host/attention_to_ffn_def.cpp#L19)
- [当前 A3/A5 注册及 HcclGroup](mc2/attention_to_ffn/op_host/attention_to_ffn_def.cpp#L72)
- [当前 proto](mc2/attention_to_ffn/op_graph/attention_to_ffn_proto.h#L42)

V3 不注册旧的 KFC/MC2 GenerateTask。当前 [AttentionToFFN GenerateTask](mc2/attention_to_ffn/op_graph/attention_to_ffn_gen_task.cpp#L33) 是隐式 `group → HcclOpResParam` 路径的一部分，只保留给旧算子。

## 8. Tiling 改造

参照 Dispatch/Combine，先把当前业务 tiling 抽成可配置的公共实现，再由 V2/V3 传不同的输入和属性索引。

建议配置项至少包含：

```cpp
struct AttentionToFFNTilingConfig {
    uint32_t contextIndex;
    uint32_t xIndex;
    uint32_t sessionIdIndex;
    // 其他输入索引
    uint32_t worldSizeAttrIndex;
    uint32_t cclBufferSizeAttrIndex;
    bool isExplicitContext;
};
```


attention_to_ffn_tiling_base.h
  → AttentionToFFNTilingConfig
  → 公共 tiling 函数声明

attention_to_ffn_tiling.cpp
  → 公共 tiling 实现
  → 旧 AttentionToFFN 注册

attention_to_ffn_v3_tiling.cpp
  → V3 索引配置
  → AttentionToFFNV3 注册

新 attention_to_ffn_v3_tiling.cpp
  → context 是第 0 个输入
  → 没有 group
  → 不调用 SetHcommCfg()
  → 从 ccl_buffer_size 检查窗口大小
  → 注册 IMPL_OP_OPTILING(AttentionToFFNV3) √

进度：做完AttentionToFFNTilingV3FuncBase和AttentionToFFNTV3ilingFunc的关系了

旧路径：

- 继续读取 `group`；
- 继续调用 `SetHcommCfg(group)`；
- 输入索引和行为不变。

V3 路径：

- 校验 `context` 是非空的 `INT32/ND` 一维 Tensor，字节数必须精确等于 `sizeof(AttentionFfnContext)`；不能让更大但 ABI 不兼容的 `Mc2MoeContext` 通过；
- 不读取 `group`；
- 不调用 [SetHcommCfg](mc2/attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling.cpp#L794)；
- 使用 `ccl_buffer_size` 校验显式 Context 对应的真实窗口容量；
- 复用 shape、量化、同步、active mask、tiling key 和 workspace 计算。

窗口所需字节数必须与 Kernel 的 512B 对齐一致：

```cpp
requiredBytes = AlignUp(tokenInfoBytes, 512) + tokenDataBytes;
```

不能直接使用当前 [未计 512B 对齐空洞的求和](mc2/attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling.cpp#L771) 做精确字节检查。

当前 tiling struct 前面仍有 `Mc2InitTiling` 和 `Mc2CcTiling`：[AttentionToFFNTilingData](mc2/attention_to_ffn/op_kernel/attention_to_ffn_tiling.h#L47)。第一阶段可以保持结构布局但在 V3 中不填这两项，避免 Host/Kernel 的 `attentionToFFNInfo` offset 不一致；清理无用字段应单独进行。

Tiling 只能检查 Context Tensor 的描述信息，不能同步读取 Device POD 中的 `rankId/rankSize/windowAddr`。这些字段由唯一合法 Builder 在创建时校验；Kernel 还应对 `rankId < rankSize`、`rankSize == worldSize` 和即将访问的地址非零做防御或异常上报。若允许外部自行构造任意 Context Tensor，这些条件就是调用契约，不能声称 Host tiling 已完成内容校验。

## 9. Kernel 改造：直接解释 Context，不加 Holder

当前所有隐式 HCCL Context 依赖集中在一份核心文件中：

- [`winContext_` 成员](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L164)
- [`GetHcclContext()` 和 rank 获取](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L182)
- [两处远端地址解析之一](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L417)
- [第二处远端地址解析](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L577)
- [本地窗口地址解析](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L598)

建议为共享核心增加编译期 `UseExplicitContext` 模式，或增加独立的 V3 Init；不要增加运行时 ContextHolder。

V3 初始化：

```cpp
context_ = (__gm__ Mc2Aclnn::AttentionFfnContext *)context;
rankId_ = context_->rankId;
```

统一地址函数：

```cpp
__aicore__ inline GM_ADDR GetWindowBase(uint32_t rank)
{
    if constexpr (UseExplicitContext) {
        return (GM_ADDR)context_->windowAddr[rank];
    } else {
        // 保留当前 HcclOpResParam 的 userMemType 分支
    }
}
```

然后仅替换上面的三处地址解析，后面的 TokenInfo/TokenData offset 和 DataCopy 逻辑不变。

新增 A5 V3 入口：

```cpp
__global__ __aicore__ void attention_to_ffn_v3(
    GM_ADDR context,
    GM_ADDR x,
    ...,
    GM_ADDR workspace,
    GM_ADDR tiling);
```

可对照当前 [A5 入口没有 Context 的签名](mc2/attention_to_ffn/op_kernel/arch35/attention_to_ffn_apt.cpp#L29)。V3 编译产物中不应再引用 `GetHcclContext()`。

## 10. ACLNN 实施顺序

### 10.1 显式 V3 API 必须与 Context 生产入口一起交付

专用 Context 不能由调用者随便拼一个 INT32 Tensor，也不能使用现有 MoE/Engram Context。发布显式 V3 API 时，必须同时提供唯一受支持的生产入口，例如：

```text
AttentionFfnContextManager(group)
  → 调用专用 Builder
  → 返回 context Tensor + cclBufferSize
  → 持有 Engine/device Context 的生命周期
```

该入口可落在公共 C/C++ setup API，或 Torch Extension 的 ContextManager 中；最终形态按正式接口规格确定。若第一阶段只有旧 API 的内部兼容 wrapper 能调用 Builder，则 `aclnnAttentionToFFNV3` 只能作为内部 inner/测试接口，不能宣称是可独立调用的公开接口。

```text
aclnnAttentionToFFNV3GetWorkspaceSize(context, ..., cclBufferSize, ...)
  → 校验 context
  → aclnnInnerAttentionToFFNV3GetWorkspaceSize(...)

aclnnAttentionToFFNV3(...)
  → aclnnInnerAttentionToFFNV3(...)
```

V3 执行段不再调用当前旧 API 中的 [NnopbaseSetHcclServerType(...MTE)](mc2/attention_to_ffn/op_api/aclnn_attention_to_ffn.cpp#L90)，因为 V3 不需要 KFC server 注入隐式 HCCL Context。

### 10.2 再做旧 API 的 A5 兼容路由

显式 V3 多卡跑通后，再参考 Dispatch/Combine 把旧 API 改成：

```text
A5 && HCOMM 版本支持 Channel Context
   && 当前为受支持的 MTE/window 类型
   && 专用 Builder 可产生等价 raw window
  → group
  → AttentionFfnContextBuilder 获取/创建 Engine Context 并包装 Tensor
  → inner AttentionToFFNV3

其他情况（A3、HCOMM 版本不足、UserMem/窗口类型暂不支持等）
  → 原 inner AttentionToFFN
```

旧 API 的二段执行函数需要在 executor 中保存实际走的是 V2 还是 V3，再选择对应 inner execute；不要只在 GetWorkspaceSize 阶段切换。

## 11. 推荐开发顺序

1. **地址与可访问性验证**：确认专用 Builder 能得到与旧路径相同的 local/remote raw window base，并完成 V3-only 冷启动跨 rank MTE 验证。
2. **冻结 ABI**：建立公共 `attention_ffn_context.h`，完成 `sizeof/offset/Tensor shape` 单测。
3. **实现专用 Builder**：创建/查询 Engine Context、拷贝 POD、按调用包装 Tensor，返回所有 rank 共同安全的 `ccl_buffer_size`；此时不申请 Engram 式外挂内存。
4. **改共享 Kernel 核心**：加入 V3 direct-cast 分支和统一 `GetWindowBase()`，保持 V2 分支原样。
5. **新增 V3 kernel 入口与 Kernel UT**：用伪造 Context 验证三个地址访问点和所有业务 offset。
6. **新增 V3 OpDef/proto/infershape/tiling**：只注册 A5，V3 tiling 跳过 group/HcommCfg。
7. **新增显式 ACLNN V3 API 与 Context 生产入口**：两者必须一起可用；否则先保持为内部接口。
8. **多卡联调**：覆盖 rank 0 和非 0、同步/异步、量化/非量化、active mask、窗口边界。
9. **最后做兼容与上层**：按能力门禁增加旧 API A5 自动路由，再做 Torch Extension、Python schema/meta、FX Converter、GE 图模式。
10. **另立 FFNToAttention A5/V3 任务**：[当前 FFNToAttention 只注册 ascend910_93](mc2/ffn_to_attention/op_host/ffn_to_attention_def.cpp#L73)，不能把 A5 往返测试作为本阶段完成条件。

这个顺序先证明最容易出错的“地址是否相同”，再铺算子接口，避免写完 Torch/图模式后才发现底层窗口基址不兼容。

## 12. 文件清单

### 必须新增或修改

| 文件/目录 | 改动 |
|---|---|
| `mc2/common/op_kernel/attention_ffn_context.h` | 唯一 Context ABI 定义 |
| `mc2/common/op_api/` 下的专用 Builder | group → raw window addresses → Context Tensor + cclBufferSize |
| 公共 setup API 或 Torch ContextManager | 为公开显式 V3 提供唯一合法的 Context 生产入口；若暂不公开则可后置 |
| [attention_to_ffn.h](mc2/attention_to_ffn/op_kernel/attention_to_ffn.h#L1) | 增加 V3 direct-cast 和统一窗口地址函数 |
| `mc2/attention_to_ffn_v3/op_kernel/arch35/` | 新增 A5 V3 kernel 入口 |
| `mc2/attention_to_ffn_v3/op_host/` | V3 OpDef、infershape、tiling 与构建配置 |
| `mc2/attention_to_ffn_v3/op_graph/` | V3 proto；不注册旧 GenerateTask |
| `mc2/attention_to_ffn_v3/op_api/` | 显式 ACLNN V3 API |
| `mc2/attention_to_ffn_v3/tests/` | ABI、tiling、kernel、ACLNN、多卡测试 |
| 构建/发布清单 | 按仓库生成链路更新 A5 binary metadata、simplified key、API 与 CI 列表（若构建要求） |

### 第一阶段保持不动

| 文件/范围 | 原因 |
|---|---|
| [旧 AttentionToFFN OpDef](mc2/attention_to_ffn/op_host/attention_to_ffn_def.cpp#L19) | 保留 A3 和旧 group schema |
| [旧 GenerateTask](mc2/attention_to_ffn/op_graph/attention_to_ffn_gen_task.cpp#L33) | 仍服务旧隐式 HCCL Context 路径 |
| [旧 A3 kernel](mc2/attention_to_ffn/op_kernel/arch22/attention_to_ffn_a3.cpp#L1) | 本任务只新增 A5 V3 |
| [旧 ACLNN API](mc2/attention_to_ffn/op_api/aclnn_attention_to_ffn.cpp#L72) | 等显式 V3 稳定后再加兼容路由 |
| TokenInfo/TokenData 布局 | Context 切换不应改变业务通信协议 |

## 13. 测试与完成标准

### 单测

- Context ABI 大小、字段 offset、Tensor 长度；
- context 的 dtype、format、shape、空指针错误；
- Builder 对 `rankId/rankSize/world_size` 的契约校验，以及 Kernel 的防御检查；
- Host 对 `ffnStartRankId/world_size` 等普通属性的范围校验；
- `requiredBytes = AlignUp(infoBytes, 512) + dataBytes` 的边界；
- V3 tiling 不读取 group、不调用 `SetHcommCfg`；
- V2 tiling 和 A3 回归不变；
- V3 kernel 的 local/remote base 及三个业务访问点。

### 多卡

- rank 0 和非 0 的地址都正确，防止 `i * 1024` 问题只在非 0 rank 暴露；
- V3-only 冷启动，不依赖旧 MC2/KFC 路径提前创建资源；
- 默认 HCCL window；如宣称支持，再覆盖 UserMem；
- 同步/异步、量化/非量化、active mask；
- 窗口刚好够和少 1B 的边界；
- Context Engine cache 重用、每次 Tensor 包装和 executor 生命周期正确。

完成标准：A5 V3 从 Op schema 到 Kernel 全程只通过显式 Context Tensor 获取通信窗口，不调用 `GetHcclContext()`；在未先运行旧算子的冷启动环境中也能完成真实跨 rank MTE 通信；旧 A3/旧 API 行为不变；多卡访问地址与旧路径逐 rank 等价。`FFNToAttention` A5 往返属于后续配对迁移的完成标准。
