# BSA MXFP8 s1 流水调优知识库

给后续生成 Cursor skill 用的原料。**本文是 reference**，不是 SKILL.md：生成 skill 时把 §0–§2、§4、§8、§9 压进 `SKILL.md`（<200 行），本文整体链成 `reference.md`。

记录截止：2026-09-12。平台 3510 / arch35 / DAV_3510。算子 `BlockSparseAttention`，tiling key `ef6ebb339b3aae0b446b8c915bf174e2`。

---

## 0. 生成 skill 时的元数据

```yaml
name: bsa-mxfp8-s1-pipeline-tune
description: >
  BSA MXFP8 arch35 s1 分区流水调优：msprof keep=1 Duration、AIC scalar 信封、
  L0C preload、pairKScaleSlot_/holdForScale、CCE 成员布局、安装目录隔离。
  Use when: MXFP8 s1_part、L0C preload、Wait(V1_C1)、QK256、K_PAIR256、
  mte1/mte2 带宽、15.07/14.94、pairKScaleSlot、holdForScale。
```

**触发**：用户改 MXFP8 s1 kernel 流水、对比 keep=1 Duration、问「为啥 15.07 vs 14.94」、动 QK/PV L1 槽或 `Wait(V1_C1)`。

**不触发**：精度-only、FP16、MXFP4、P0B 基线对比（除非用户明确拿 s1 对照）。

**与已有 skill 分工**：

| skill | 管什么 | 本文补什么 |
|---|---|---|
| `msprof-op-perf` | 怎么采、CSV 字段、达标阈值 | s1 专用判读：Duration≈aic_scalar，bw 不是墙 |
| `ascendc-perf-optimize` | 卡间/核间/核内通用流程 | s1 已验证杠杆与禁区 |
| `ascendc-operator-compile-debug` | 通用编译测试 | s1 的 cann-9.1.0 编译 / beta.1 上板、OPP overlay |

`disable-model-invocation` 建议 `false`（关键词多，需要自动挂上）；若怕误触发则 true，靠用户 @ skill。

---

## 1. 硬件与活路径

- L0C 256KB，L0A 64KB，UB 248KB。
- QK 与 PV **共用** `kvBufId`（L1 KV 4 槽）和 `KV_EVENT0..3`。
- 活 flags（默认 1，勿在不知情时关掉）：

| 宏 | 作用 |
|---|---|
| `BSA_MXFP8_K_PREFETCH` | Wait 前 PrefetchK（P0b） |
| `BSA_MXFP8_K_PAIR256` | 偶拍 gather 256 行 K，奇拍复用下半 |
| `BSA_MXFP8_QK_MMAD256` | 偶拍一次 mmad M=256，两次 Fixpipe 128 |
| `BSA_MXFP8_NDNUM2` | 等长非连续两块一条 Nd2Nz |
| `BSA_MXFP8_L0C_PRELOAD` | 偶拍 mmad 进同一块 128KB L0C S，Wait 挪到 Fixpipe 前 |
| `PV_RE_DELAY_N=20` | PV 相对 QK 延迟 20 拍 |

L0 规划（utils）：K 32KB@L0A0、Vᵀ 18KB@L0A32；QK L0C 128KB@0 跨 even/odd 两次 Fixpipe；PV L0C 72KB@128。不新开第二块 L0C S。

偶拍（preload on）：

```
PrefetchK → ComputeQKMmad256Even → Wait(V1_C1, PIPE_FIX) → EmitFixpipeQK256Even → SetFlag → … → PV
```

奇拍：Wait 仍在 Fixpipe1 前（UB ping-pong）。奇拍没有 QK mmad。

---

## 2. 测量契约（违反则数字作废）

**点**：`S=118800 keep=1.0`，`warm-up=5 launch-count=10`，`BSA_OUT=bfloat16`。
脚本：`tests/mxfp8_pytest/run_msprof_points.sh` + `report_msprof.py`。
冒烟：`run_golden_regression.sh`，至少 `qs64_kvs128 / qs128_kvs128 / qs128_kvs256`。

**环境**：

- 编译：`source /usr/local/Ascend/cann/set_env.sh`（9.1.0）。
- 上板/msprof：`source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh`。
- `/usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini` 常 750 root。用 overlay
  `/home/icebearPolar/tmp/cann_opp_readable`（`load_priority=custom_nn,custom_transformer`），
  **在** `source cann/set_env.sh` **之后** `export ASCEND_OPP_PATH=...`。
- `cp -a` 出的 install 里 `set_env.bash` 可能仍指向 bootstrap。脚本必须再
  `export ASCEND_CUSTOM_OPP_PATH=${INSTALL}/vendors/custom_transformer`。

**卡**：Health OK + 空闲（或仅很小 atk）。Warning 空卡（npu0 曾 DeviceProf hang rc=143）不要当正式点。跨卡对比只认 cube/instr 是否相同；Duration 差 <20 µs 当噪声。

**墙钟**：`OpBasicInfo.csv` 的 `Task Duration(us)`（`report_msprof.py` 的 `dur`）。
**不要**用仿真 span 当 Duration。

**隔离**：每次实验 **新 install 目录**。禁止覆盖：

- `cann_bsa_0909_s1_part`
- `cann_bsa_0909_alongq`
- `cann_bsa_0909_s3`
- `cann_bsa_0827_k_prefetch_p0b`
- `cann_bsa_fp8_512_zn`

编译：rsync `op_kernel/` → `build/binary/ascend950/src/block_sparse_attention/op_kernel/`，
`BlockSparseAttention-block_sparse_attention-3.sh --kernel-src=...`，把新 `.o/.json` 拷进 install。
源码变了但 md5 不变 → CCE 消掉了；不要空跑 keep=1。改成员/寻址时建议 `rm -rf kernel_meta_BlockSparseAttention_ef6ebb*`。

不提交、不改 git config，除非用户明确要求。

---

## 3. Profiler 判读（s1 专用）

`report_msprof.py` 取 PipeUtilization **列 max**（通常 cube0 block 较大的那行）。

| 字段 | s1 上的含义 |
|---|---|
| `dur` ≈ `aic` ≈ `aic_scalar_time` | 墙钟 = AIC scalar 信封。`WaitFlag` / `CrossCoreWaitFlag` 算 scalar ISS |
| `cube` | MMAD 忙时。s1 keep=1 **锁死 10406.2**；变了才说明算子变了 |
| `vec` | AIV VF 忙时。约 11164。墙钟−vec ≈ 握手税 |
| `mte2_instr` / `read_KB` | 搬运工作量。s1 keep=1 **127447 / 1042182** |
| `mte2_time` / `mte2_bw` | 同一批字节的占用与带宽。bw 升通常是 **issue 更早/更紧**，不是多 copy |
| `mte1` | LoadK/LoadV 忙时。升高且 Duration 下降 = 叠进了 Wait 窗口 |

判据：

1. cube/vec/instr 不变、Duration 变 → **scalar 信封**（Wait 何时发、发完后还有没有串行尾巴）。
2. mte2_bw 92 vs 97 且 instr/readKB 相同 → 不是少搬，是 MTE2 busy 被拉长。
3. MTE2 time ≪ Duration 时，再抬 bw **不会**掉墙钟。
4. `aic_scalar_ratio` 0.99 → AIC 几乎一直在等旗；优化对象是把活塞进等旗窗口，或缩短等旗后的尾巴。

---

## 4. 基线包（keep=1 S=118800）

| 包 | `.o` md5 | size | Duration | 说明 |
|---|---|---|---|---|
| Sep 9 真源 / `s1_sep9` / `s1_clean` | `a6293193` | 1693424 | **15.05–15.08** | 现场 s1，**未进 git** |
| `55c9dcb3d` 脏 s1（无 preload） | `5239f944` | 1692912 | **15.21–15.24** | git 第一次出现 QK256，顺手内联了 scale 槽 |
| `240751a27` 恢复 | `a6293193` | 1693424 | 15.07 | 把 Sep 9 QK/PV live path 加回 git |
| git `d3caa0fb6` | `653fe4b5` | — | **17.99** | L0A 16KB / L0C 64KB，**不是** 15.06 |
| 0912：s1 槽 + preload | `42727d7a` | 1697264 | **15.07** | Wait 已后移，成员二次索引把窗口吐出来 |
| 0911 / 0912_scaleslot：直传槽 + preload | `f02a5af2` | 1696752 | **14.93–14.95** | 与脏+preload **逐字节相同** |
| P0B `k_prefetch_p0b` | `b19…` | — | ~16.54 | 对照用，勿当 s1 目标 |

15.06 包无法从 git 复原。不要把 d3caa 或 s1b 当成 15.06。

---

## 5. 已验证因果

### 5.1 墙钟是 AIC 等 `V1_C1`，不是多打 MMAD/VF

四组 cube=10406.2、instr=127447。`Duration − max(cube,vec) ≈ 3.8–3.9 ms` 是重叠气泡。脏无 preload 多出的 0.14 ms 是更厚的 scalar 墙，不是注释或 `#if 0`。

### 5.2 「脏」是减代码，「s1_part」才带骨架

`55c9dcb3d` 净删（QK/PV/utils +127/−209）。两类删除：

- 注释 / 关死的宏：进不了 `.o`，与 Duration 无关。
- **live C++**：`pairKScaleSlot_` 成员、`pairKScaleSlot_ = pairKSlot_`、`holdForScale=false` 的 `if`、PV 对称的 `pairVScale*`。

Sep 9 s1_part 不是「减宏后的干净树」，是当时能编出 `a6293193` 的现场源，留着 **未打开的 hold-scale 分槽骨架**（功能关死但表达式仍编译）。`240751a27` 加回骨架是为了对齐 15.07 二进制，不是因为更干净。

`SparseKFused` 里 `(void)kBufId`，真正进 `copyGmToL1MxScaleDn2Nz` 的是 `l1AScaleTensor[scaleBufId]`。运行值永远 `scaleBufId == pairKSlot_`，但走 **class 成员** 时 CCE 不能当成同一寄存器：多 512 B、每段 gather 间接寻址。

### 5.3 同一处内联，有无 preload 方向相反

无 preload：`Wait → LoadK → SetFlag×2 → mmad → Fix`。骨架里的 `if (!holdForScale)` 在 Wait **后** 给 SetFlag 留间隔 → 15.07；内联背靠背 SetFlag → 15.22。

有 preload：`LoadK → SetFlag → mmad → Wait → Fix`。内联让 LoadK/mmad 更早塞进 AIV softmax 窗口 → 14.94；留下成员则 mmad 探出窗口，Fixpipe 晚 0.13 ms → 15.07。

`holdForScale` 的 `if` 和未使用的 `nextOriYBlockIdx_`：**CCE 消掉**，md5 不变。只有 **被写入并作为下标使用的成员** 造成 0912↔0911 的 512 B。

### 5.4 偶拍时间轴（相对刻度）

灰底 = AIV 仍在做上一槽 softmax。虚线之后的 AIC 活 1:1 进 Duration。

```
AIV softmax        |========================|
脏+preload MTE1/mmad      |LoadK|mmad|Wait|Fix0   Fix 贴虚线 → 14.94
s1槽+preload MTE1/mmad         |LoadK|mmad |W|Fix0  mmad 越线 → 15.07
```

mte1 7154 vs 7030、bw 96–97 vs 92.4 = 同一批 copy 有没有叠进灰底，不是脏路径更能打满总线。

### 5.5 在 s1 语义下拿到 14.94 的最小 diff

热路径 **直传** `pairKSlot_` / `vSlot`，不要 `pairKScaleSlot_` / `pairVScale*` 成员。
当前工作区 + `cann_bsa_0912_l0c_scaleslot` 已是此版，md5=`f02a5af2`，npu4 **14.952 ms**。
若将来真分槽：用 **局部变量** `uint32_t scaleBufId = pairKSlot_`，禁止进 class。

---

## 6. CCE / 源码陷阱

1. 注释、`#if 0`、值为 0 且无 `#else` 活体的宏 → 无 `.o` 差。
2. `bool x=false; if(!x) SetFlag` → 通常被折掉（偶拍已验证 md5 不变）。
3. **未读死成员**插在 `kBufId` 和 `qkL0abBufId` 之间仍可能改布局；`nextOriYBlockIdx_` 这次被 DCE，**不要赌下一次**。
4. 恒等赋值进 **另一成员再当数组下标** → 几乎一定多代码。`pairKScaleSlot_=pairKSlot_` 就是。
5. 改完先 `md5sum` `.o`。与对照包相同则不要上板。
6. Host `torch_extension` 改不动 kernel Duration。

---

## 7. 优化杠杆（按层）

已打到 14.94 后，MTE2 不在墙上（busy ~10.4 ms ≪ 14.9）。再抬 bw 无 Duration。剩余税 `Duration−vec≈3.8 ms`。

| 优先级 | 杠杆 | 预期 | 状态 |
|---|---|---|---|
| 0 | 热路径直传槽 + L0C preload | 15.07→14.94 | **已验证** |
| 1 | 偶拍 Wait 前 PrefetchV（仿 PrefetchK） | 缩短 Wait 后 PV 尾巴 | 未做；L1 4 槽，K 槽已在 even LoadK 后释放 |
| 2 | 奇拍 Wait 前预取下一偶拍 K | 填空的奇拍 Wait | 未做；环形槽与 PV 占用要算清 |
| — | 第二块 L0C S | 与 128KB 同槽设计冲突 | 不做 |
| — | 奇拍 Wait 挪到 Fixpipe1 后 | UB ping-pong 会打 | 不做 |

---

## 8. 禁止重试（已证伪或会毁掉基线）

- 把 sim span 当 Duration。
- 把 0911 的 −0.30 vs 脏 15.24 说成 vs 15.07；真值是 **−0.13 vs 15.07**。
- 用 d3caa / s1b 当 15.06。
- alongQ qs128 VF、QK split、VF ACC、mux、拿 s3 当 perf 包。
- 无 preload 时再简化 `holdForScale`（15.07→15.22）。
- 有 preload 时把 `pairKScaleSlot_` 加回 **class 成员**（14.94→15.07）。
- 覆盖 §2 列出的 install 目录。
- 为「清理」而删 live 成员却不上板 keep=1。

---

## 9. Agent 工作流（生成 SKILL.md 时原样保留）

1. 读本文 §2、§4，确认对照包 `.o` md5。
2. 改 kernel 前写清：动的是 Wait 前还是 Wait 后（方向会反）。
3. 新目录编译；对比 md5/size；变了再冒烟 3 点，再空卡 keep=1。
4. 汇报必须带：Duration、cube、vec、scalar、mte1、mte2_bw、instr、md5、卡号。
5. cube 或 instr 变了先查是否误关 flag / 误改 mmad，再谈流水。
6. Duration 掉了但 bw 没动：可以，墙钟跟 scalar。bw 涨了 Duration 不动：MTE2 不在墙上，换杠杆。

---

## 10. 编译命令骨架

```bash
OPP_FIX=/home/icebearPolar/tmp/cann_opp_readable
INSTALL=/home/icebearPolar/cann_bsa_YYYYMMDD_<tag>   # 从 s1_part cp -a，勿覆盖
# rsync op_kernel → build/binary/ascend950/src/block_sparse_attention/op_kernel/
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_OPP_PATH="$OPP_FIX"
# BlockSparseAttention-block_sparse_attention-3.sh --kernel-src=... py bin/
# cp 新 .o/.json → $INSTALL/.../ascend950/block_sparse_attention/
```

上板：

```bash
INSTALL=... TAG=... ASCEND_RT_VISIBLE_DEVICES=<ok空卡> \
  S_LIST=118800 KEEP_LIST=1.0 \
  bash tests/mxfp8_pytest/run_msprof_points.sh
```

---

## 11. 关键源文件

- `op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` — PrefetchK / Wait / Fixpipe 编排
- `op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp` — PrefetchK、Mmad256Even、SparseKFused `scaleBufId`
- `op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp` — V gather / LoadV 槽
- `op_kernel/attn_infra/gemm/block/block_mmad_arch35_utils.hpp` — flags 与 L0/L1 尺寸
- `tests/mxfp8_pytest/run_msprof_points.sh`、`report_msprof.py`、`run_golden_regression.sh`

对照图（偶拍时间轴）：工作区 canvas
`~/.cursor/projects/home-icebearPolar/canvases/s1-part-vs-dirty-preload-even-tile.canvas.tsx`
