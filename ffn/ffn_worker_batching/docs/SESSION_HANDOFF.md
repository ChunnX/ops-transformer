# FfnWorkerBatching A5 开发 — 会话恢复锚点（2026-07-08 存盘）

> 上个会话在 review 后主动存盘、用户重启会话。**新会话恢复请先读本文件 + LOG.md。**

## 一句话状态
FfnWorkerBatching 新增 Ascend950(arch35/A5) 支持：**开发已完整收口 + 代码检视已完成**，下一步是 **changwei-op-dev 流水的 step4 出包安装 → step5-8 泛化验证（边验边修）→ 性能/精度 → 上库**。

## ⚠️ 采用的工作流（重要，别搞错）
- 本任务用 **`changwei-op-dev` 手册**（`/home/zl/changwei-op-dev/SKILL.md`）主导，**不是** CLAUDE.md 默认的 `ops-registry-invoke-workflow`（用户明确切换）。
- changwei 流水：1定架构+写算子 → 2交付物 → **3代码检视 → 4出包安装** → 5生成用例 → 6抽检 → 7批跑low → 8full泛化(出功能xlsx) → 9性能xlsx → 10精度xlsx。
- 写代码环节加载子 skill `ascendc-dev`；检视用 `ascendc-code-review`；泛化用 TTK + `ascendc-whitebox-design`。
- **第一铁律**：验证前必须重新出包+装，泛化跑部署包（非 build/ 产物/旧包）；每轮 TTK 前 `rm -rf ops-test-kit/kernel_meta/dyn_op_*`。

## 环境 / 路径
- 算子目录：`/home/zl/ops-transformer/ffn/ffn_worker_batching/`
- CANN 环境：`source /home/zl/Ascend/ascend-toolkit/set_env.sh`（不用 /root/Ascend）
- TTK：`/home/zl/ops-test-kit`（本算子唯一输入 schedule_context 内嵌 GM 绝对地址，**TTK kernel 模式无法直驱** → kernel 级验证走 Kernel直调，TTK 用于 aclnn/E2E 层）
- 设备：Ascend950DT_9584（arch35/dav-3510）可用
- 分支：`operators/ffn_worker_batching`；tag 到 `design-approved`

## 已完成
- **阶段一 需求/设计**：全 ✅（REQUIREMENTS/spec.yaml(9-stage PASS)/DESIGN/PLAN/TEST + 三评审全过）。
- **完整开发收口**（dev subagent 产出）：arch35 是 A2 忠实全量移植（15 kernel 头与 A2 逐字节一致，仅 2 处必要差异），无桩；NORM(100)+RECV(101) 双 TilingKey、token_dtype 0/1/2、单多核 sort+MrgSort、边界硬校验、UB 运行时取值全接线；k=0 守卫 NORM 已落、RECV 由哨兵天然规避。补：README A5→√、6 条 arch35 边界负路径 UT。**编译**：ascend950 `--opkernel` EXIT 0 + 3 binary(15:22)；ascend910b EXIT 0 零回归；op_host UT **24/24 PASS**。全程未碰 A2/A3 分支、保持 `_apt.cpp` 接线。
- **代码检视（changwei step3）✅**：报告 `docs/arch35_code_review_summary.md`。结论：delta 无阻塞级逻辑缺陷；1 个 HIGH 待泛化实测。

## 检视关键结论（承接到验证阶段）
- **【H1 头号】UB 大 shape 溢出风险未验**：`ffn_worker_batching_tiling_arch35.cpp:220` 的 `sortLoopMaxElement` 从 host `GetCoreMemSize(UB)=248KB` 派生（=7936）。但 PROBE_SUMMARY 穿刺2 说 kernel 侧真实 216KB、4×SLM footprint=248KB 会**溢出**；INTEGRATION_REPORT 又说以 248KB 为唯一真值源。**两文档矛盾，代码走 248KB，大 shape NPU 未实测**。→ 泛化必跑大 shape(A=1024,K=64,单核 loop tile≥7936)，若 507035/踩内存则改分母用 min(host,kernel)（禁写死常量）。issue：`issues/issue_20260708_ub-size-host-kernel-mismatch_01.md`。
- **【M2】多核 group_listing k=0 未 NPU 覆盖**（单核已守卫验证；多核靠 count>0 过滤天然产[0,0]，未实测）→ 泛化补多核 k=0 用例。issue：`issues/issue_20260708_group-listing-kzero-dirty-expertid_01.md`（单核 fix 已落主线 arch35 头）。
- 全 dtype×2路径×边界 的 NPU 全量对拍 A2 仅 probe 抽样过，泛化补全。
- RECV `scan_token_info.h:100` while(true) 忙等无超时 = 分离部署协议固有设计(A2一致)，非缺陷。

## 下一步（新会话直接做）
1. **step4 出包安装**：先 `python3 /home/zl/changwei-op-dev/scripts/check_arch35_config.py`（自查 ascendc_config.json 有 ascend950）→ `cd /home/zl/ops-transformer && bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching --vendor_name=custom -j32` → `./*.run --quiet` merge 安装。
2. **step5-8 泛化验证（边验边修，用户已授权）**：whitebox 生成泛化CSV+golden → 抽检2-3真实case(含 H1 大shape、M2 多核k=0) → 批跑 low → full 全泛化(TTK 后台、分块、清 kernel_meta 缓存) → 出功能 xlsx（判据 REAL_FAIL=0）。
3. step9 性能 xlsx / step10 精度 xlsx → step 上库(README/文档/总结报告)。

## ⚠️ 两个悬项（需用户拍板或注意）
1. **全部代码未 commit、未 tag**：git 停在阶段 1.4（`c6c07d462`）。迭代一/二 + 完整开发收口 + review 的产物（arch35 kernel、apt.cpp、tiling_arch35、config/ascend950、probe/、ST、6条新UT、README、review报告 等 20+ 项）**全在工作区未提交**。**再崩会丢**。建议新会话尽早补 checkpoint commit。
2. **arch35 接线约定**：现用 `_apt.cpp`（符合 memory `ops-transformer-arch35-convention` + 现有代码）；changwei-op-dev §56 要求 `add_kernel_sources`。用户已**暂时深priori化**（先完整开发+验证），若最终强制 changwei 约定则接线要返工。

## 看门狗
session-only cron `4f0599be`（每3分钟自检卡死）——重启会话即失效，已在存盘时删除。
