# M-RoPE MX `simt_vk` 隔离实验设计

> 状态：历史实验记录。该实验曾完成目标 shape 的 clean package、路由、精度和 matched 性能验证；
> 当前源码已删除私有 key、路由和 SIMT/SIMT-VK 实现文件，本文只保留历史方案与证据，不代表当前交付实现。
> 公开输入输出以 `docs/srs_stc/mrope_mx_srs.md` 为准。

## 0. 与当前实现冲突或必须先确认的点

| 项目 | 实验设计 | 影响 |
|---|---|---|
| 新 tiling key | 在八维模板末尾的 `EXEC_MODE` 增加 `SIMT_VK=2`；实际 key 为 `8891994368`。 | 现有 Vector `302059776`、SIMT `4597027072` 的数值和选择条件保持不变。 |
| 私有选择值 | `QKV_MROPE_MX_IMPL=simt_vk` 只对 M-RoPE+MX scene 生效。 | 不新增公开 attr、ACLNN 参数或 Torch 参数；非法值与现有行为一样只拒绝新 scene。 |
| V/K 顺序 | 当前 SIMT 是全核 `V phase -> K phase -> Q phase`；实验改为每个 token、每个 head-pair 内 `V -> K`，全部 VK 完成后再执行 Q。 | 不再保持“所有 V 完成后才开始任一 K”的全局阶段顺序；V/K 互不读写且输出集合不变，因此数值合同不变，但流水身份必须单独验证。 |
| 新实现实体 | 新增 `mrope_mx_simt_vk.h`，只拥有 VK 合并调度；RMSNorm、M-RoPE、MX scale、FP8 转换和 store helper 继续复用现有 SIMT 事实源。 | 防止复制数值公式形成漂移；新文件不是旧实现的转发别名。 |
| 资源与同步 | 仍为 AIV-only、GM 直读直写、寄存器中间值；不使用 UB staging、L1/L0、workspace 或跨核同步。 | 两次 `asc_vf_call` 按 `VK -> Q` 提交，末尾保留一次 `V_S` 完成 fence。 |
| 证据身份 | 新 selector 会生成新的编译产物。 | 旧 object 的精度、性能和 profiling 不能证明新 key；旧 key 需做路由回归，新 key 需独立编译、精度和性能证据。 |

## 1. 为什么需要独立 key

当前 SIMT 在目标 shape `T=768,Nq=64,Nk=Nv=8,D=128` 上约 `53.85~54.11 us`，whole-kernel AIV Vector 时间约占 `95.93%`。线程数、循环展开、scale 写回、整数 scale 和 K/Q 调用融合均未产生稳定收益。

VK 合并改变了任务映射和阶段边界，不能作为现有 SIMT key 内的无感微调。独立 key 提供三个隔离保证：

1. 默认 Vector 与已有 SIMT 的选择和值不变；
2. 新实现可通过实际 kernel task name/key 单独证明；
3. 若实验回退，可删除一条 selector、一个 Controller 特化和一个实现文件，不影响现有路径。

## 2. Host 路由

```text
QKV_MROPE_MX_IMPL unset/vector -> EXEC_MODE=0 -> key 302059776
QKV_MROPE_MX_IMPL simt         -> EXEC_MODE=1 -> key 4597027072
QKV_MROPE_MX_IMPL simt_vk      -> EXEC_MODE=2 -> key 8891994368
```

三个值必须在独立进程中比较，避免 framework tiling/kernel cache 复用旧选择。新 key 与已有两个 key 共用同一公开合同、TilingData ABI、按 token 商余分核和 AIV-only blockDim。

## 3. 核内任务映射

每个 AIV 仍获得连续 token 区间 `[tokenBegin, tokenEnd)`。VK VF 使用 `512 threads = 16 warps`，一个 warp 在一次迭代中拥有一个 token：

```text
tokenIndex = warp + n * 16
lane 0..15  -> subgroup 0 -> headPair 的偶数 head
lane 16..31 -> subgroup 1 -> headPair 的奇数 head
subLane 0..15 每 lane 处理四个 D32 block 中的相邻两个元素
```

对该 token：

1. 读取一次 `slotMapping[token]`，计算一次 cache block/token base；
2. 读取一次三轴 `mropePosition[token,:]` 并形成该 token 的 RoPE factors；
3. `headPair=0,2,...` 遍历 K/V heads；每对 head 先完成 V 静态量化写回，再完成 K RMSNorm+M-RoPE+MX 写回；
4. K scale 继续按每 head 四个 E8M0 byte 写入，保持对齐 fast path 与 byte fallback；
5. VK VF 全部提交后，现有 Q VF 按原映射和数值 helper 执行。

与已有 SIMT 相比，实验删除 V/K 之间一次 VF 提交，并把 V 中每个 head-pair 重复的 slot 商余解码收敛为每 token 一次。代价是目标 shape 每核 12 token 时 VK 阶段只有 12 个活跃 warp，且 V/K 在 token 内串行；性能结论必须由 matched A/B 决定。

## 4. 模块与唯一事实源

```text
Host ParseMropeMxExecMode("simt_vk")
  -> EXEC_MODE=2 selector (AIV-only)
  -> Controller<..., SIMT_VK>::Process
  -> ProcessMropeMxSimtVk
       -> MropeMxSimtVk VF
            -> existing MropeMxSimtLoadPairGammas
            -> existing MropeMxSimtLoadPairRopeFactors
            -> existing MropeMxSimtPairRmsRope
            -> existing MropeMxSimtPairMakeScales
            -> existing FP8/scale store helpers
       -> existing MropeMxSimtQ VF
       -> V_S completion fence
```

新文件只新增 V 静态量化与 K 处理的合并遍历。它不得复制 M-RoPE 选轴、RMSNorm、CUBLAS MX scale 或 FP8 cast 公式。

## 5. 完成门槛与结果

- Host UT：unset/`vector`/`simt`/`simt_vk` 四路 key 精确匹配，非法值失败，non-MX scene 忽略该环境变量；
- 编译：selector 生成 key `8891994368`，现有两个 key 仍存在；
- 路由：设备 task name 实际以 `_8891994368` 结尾；
- 精度：先以 `T=768,Nq=64,Nk=Nv=8,D=128` 验证五输出均 `100%`，通过后再决定是否跑完整矩阵和最大 T；
- 性能：独立进程、相同 CSV、1 次 warmup、19 次 task time，至少一组反序 matched A/B；
- 接受：只有在两个方向都快且合并收益超过噪声门槛时，才把它描述为优化；否则保留为功能实验或删除，不替换现有 SIMT 基线。

当前结果：

| Gate | 结果 |
|---|---|
| Host UT | `155/155 PASS`；unset/`vector`/`simt`/`simt_vk` 精确 key 与非法值/非 MX 场景行为覆盖 |
| clean opkernel/package | PASS；三个目标 JSON 均包含 `302059776/4597027072/8891994368` |
| release Kernel TTK 路由 | 原 SIMT 实际 `4597027072`，SIMT-VK 实际 `8891994368`；两路编译成功 |
| Kernel TTK 精度 | 两路 Nq=64 用例均五输出 `100%`，memory check 全 OK |
| ACLNN 目标 shape | 两路 `T=768,Nq=64,Nk=Nv=8,D=128` 均五输出 `100%` |
| matched 性能 | SIMT 三轮 `53.86/53.99/53.77 us`；SIMT-VK `52.50/52.61/52.39 us`；合并降低约 `2.55%` |

结论：在目标 shape 上收益跨三个相邻配对重复出现，因此可称为小幅优化；仍保留独立 key，不切换默认实现。未完成的边界是 SIMT-VK priority-0 Kernel 全矩阵、最大 T、Torch E2E 和详细 msopprof，不能把 2.55% 外推到其他 shape。

有效证据目录：

```text
/tmp/qkv_mrope_mx_simt_vk_isolated_20260814/
  precision_simt/result.csv
  precision_simt_vk/result.csv
  kernel_route_simt_control/console.log
  kernel_route_simt_vk_v5_sitecustomize/console.log
  perf_a3_simt_logged/console.log
  perf_b3_simt_vk_logged/console.log
```

Kernel TTK 的前两类结果明确拒绝：`DYN_OPERATOR_NOT_FOUND` 虽被工具汇总成 PASS，但 operator 未执行；`OPTILING_FAILURE` 找到 release object 但未产生合法 tiling。最终只在 `/tmp` 建立包目录名到 TTK 固定 `impl.dynamic` 名称的测试派生视图，未修改产品包，随后得到上述真实路由与精度结果。
