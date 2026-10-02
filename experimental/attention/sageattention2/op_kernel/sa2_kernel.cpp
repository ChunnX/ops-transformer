// sa2_kernel.cpp — SageAttention2 MIX 1 AIC:2 AIV 内核入口（DTYPE × GRAN × CAUSAL 8 特化分发）。
// 入口符号 = sa2_attention2（与 opInterface.value 一致）；op 类型 = SageAttention2（与 op_host/op_api 一致）。
#if !defined(__AICORE__) && !defined(__CCE_KT_TEST__)
#error "op_kernel/ is strictly reserved for Ascend NPU device compilation (__AICORE__ or __CCE_KT_TEST__)! Host simulation must use sim/sa2_sim_engine.h instead."
#endif

#include "sa2_spec.h"
#include "sa2_template_key.h"
#include "sa2_planner.h"
#include "sa2_ub_plan.h"
#include "sa2_events.h"
#include "sa2_kv_prep.h"
#include "sa2_attention_loop.h"

namespace sa2 {

// 拆分 pass 引擎甄别（value-dependent 谓词）：引用模板形参 D 使 if constexpr 条件依赖
// 模板实参，实例化期按当轮 pass 的 g_coreType（AIC/AIV 常量）真正弃置对侧分支。
template <typename D>
__aicore__ constexpr bool Sa2PassIsAic()
{
    return g_coreType == AscendC::AIC;
}

template <typename Dtype, bool GranPerThread, bool IsCausal>
class SageAttention2Kernel {
public:
    __aicore__ inline SageAttention2Kernel() = default;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
        GM_ADDR workspace, const Sa2TilingData *td)
    {
        q_ = q;
        k_ = k;
        v_ = v;
        out_ = out;
        lse_ = lse;
        tilingData_ = td;
        const Sa2Params &p = td->params;
        userWorkspace_ = AscendC::GetUserWorkspace(workspace);

        int64_t blockIdx = AscendC::GetBlockIdx();
        // P1-1 块语义修复（v27）：MIX 内核 GetBlockIdx = MIX 块号（同块 AIC/AIV 同号），
        // AIV 在块内以 GetSubBlockIdx ∈ {0,1} 区分——同硬件同模型的 FlashMLA 内核即此用法
        //（csrc/ascend_kernels/prefill/sparse/kernel_body.h:879 get_block_idx / :918 get_subblockid）。
        // 旧写法 coreId_ = blockIdx/2 会让相邻两块的 AIV 计算出相同 globalAivId：一半 KV head
        // 漏预处理 + 双倍重复写。真机语义最终核实仍挂 M2b 950PR 语义核实项。
        coreId_ = blockIdx;
        if (g_coreType == AscendC::AIC) {
            streamId_ = 0;
        } else {
            streamId_ = AscendC::GetSubBlockIdx();
        }

        // 统一有序任务划分（针对本 Die 内的 AIC 核均匀划分）
        int64_t pairCols = PairCols(p.hk);
        int64_t tiles = TileCount(p.qp);
        int64_t group = GroupSize(p.hq, p.hk);
        int64_t perHead = PerHead(group, tiles);
        int64_t slots = SlotsPerDie(p.b, pairCols);
        totalTasksPerDie_ = TasksPerDie(slots, perHead);

        // 正确的 Die 与核内分配推导（P0-1 闭环修复）：
        if (p.cores <= 1) {
            // 单核目标形态（DESIGN_SPEC 1 AIC）：覆盖全部 2 * totalTasksPerDie 任务，杜绝奇数 KV head 丢失
            range_ = Sa2Range{0, 2 * totalTasksPerDie_};
            assignedDie_ = 0;
        } else {
            // 多核形态：偶数核绑定 Die 0，奇数核绑定 Die 1；每 Die 拥有 (p.cores / 2) 个 AIC 核
            int64_t coresPerDie = p.cores / 2;
            int64_t coreInDie = coreId_ / 2;
            assignedDie_ = coreId_ % 2;
            range_ = SplitRange(totalTasksPerDie_, coresPerDie, coreInDie);
        }
    }

    __aicore__ inline void Process()
    {
        const Sa2Params &p = tilingData_->params;

#if defined(__NPU_ARCH__)
        // 阶段一：KV 预处理（仅 AIV 执行，48 AIV 全栅格并行，单写者无锁）
        if (g_coreType == AscendC::AIV) {
            ProcessKvPreprocessing(p);
        }
        // ⚠ M2b 实测修订（拆分编译 dav-c310-vec）：原「SyncAll 前 PipeBarrier<PIPE_ALL>
        // 排空写队列」在 AIV loop 降低中为不受支持的屏障指令（后端硬错误
        // "Unsupported scalar instruction in AIV loop"，-O0/-O3 同现，最小对照实测定位）。
        // 阶段一全部为 AIV 核内标量/UB 写 + GM 标量写，跨核可见性由下方 SyncAll 的
        // 全栅格驻留同步收口；PIPE_ALL 屏障是否仍有必要（如未来接入 MTE 写通路）挂 M2b 真机项。

        // 单屏障收口（AIC 与 AIV 共同参与全栅格驻留同步）
        AscendC::SyncAll();
#endif

        // 阶段二：注意力主循环 —— AIC 执行 Cube MMAD 引擎，AIV 执行 Vector 引擎。
        // M2b-0（拆分编译口径）：g_coreType 在 -cube/-vec 拆分 pass 下为 constexpr 常量
        //（sys_macros.h SPLIT_CORE_CUBE/VEC）。条件经 Sa2PassIsAic<Dtype>() 引入模板形参 →
        // value-dependent ⇒ if constexpr 的弃置分支在实例化期真正剔除：AIV pass 不实例化
        // Mmad/LoadData/Fixpipe 路径（后端 "Unsupported scalar instruction in AIV loop"
        // 硬错误），AIC pass 同理豁免 VF 段；单 pass 内 g_coreType 恒为本引擎，运行期行为不变。
#if defined(__NPU_ARCH__)
        // arch pass（-cube/-vec 拆分）：仅实例化本引擎分支（value-dependent 剔除对侧）。
        if constexpr (Sa2PassIsAic<Dtype>()) {
            ProcessAttentionCube(p);
        } else {
            ProcessAttentionVector(p);
        }
#else
        // MIX 驱动第三 pass（无 __NPU_ARCH__，MicroAPI/Cube 内联不可用）：只承载标量壳与
        // 符号表（对齐框架 MIX 模式 update_host_stub.py 的 host-stub 产物定位）。
        (void)p;
#endif
    }

private:
    __aicore__ inline void ProcessKvPreprocessing(const Sa2Params &p)
    {
        constexpr Gran granMode = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
        Sa2KvPreprocessor<Dtype, granMode> kvPrep;

        const int64_t chunksPerHead = (p.kp + kKvStep - 1) / kKvStep;

        if (p.cores <= 1) {
            // 单核形态（DESIGN_SPEC 1 AIC 目标）：核内双 AIV 并行
            int64_t aivId = (coreId_ >= p.cores) ? (coreId_ % 2) : streamId_;
            int64_t totalKvHeads = p.b * p.hk;

            // Pass 1: 均值预统计（无条件执行）。smooth 关闭时 ProcessK/VMeanDevice 只落零值
            //（不读 GM）：Pass 2 的 K 中心化（ProcessKQuantChunkDevice 减 wsKMean）与 AIV 侧
            // corr = q·μ（StepVectorQQuantDevice）都无条件消费 wsKMean，而 workspace 不由框架
            // 清零——按 smooth 条件跳过本 pass 会让 smooth_k=False 路径读到未初始化 μ。
            for (int64_t kvIdx = aivId; kvIdx < totalKvHeads; kvIdx += 2) {
                int64_t b = kvIdx / p.hk;
                int64_t kh = kvIdx % p.hk;
                kvPrep.ProcessHeadMeans(k_, v_, userWorkspace_, *tilingData_, b, kh);
            }
            AscendC::SyncAll();

            // Pass 2: 128 行 Token 分块网格并行量化，平铺到核内双 AIV
            int64_t totalTiles = totalKvHeads * chunksPerHead;
            for (int64_t tileIdx = aivId; tileIdx < totalTiles; tileIdx += 2) {
                int64_t kvIdx = tileIdx / chunksPerHead;
                int64_t chunkIdx = tileIdx % chunksPerHead;
                int64_t b = kvIdx / p.hk;
                int64_t kh = kvIdx % p.hk;
                int64_t chunkStart = chunkIdx * kKvStep;
                int64_t chunkLen = aicMin(kKvStep, p.kp - chunkStart);
                if (chunkLen > 0) {
                    kvPrep.ProcessHeadChunk(k_, v_, userWorkspace_, *tilingData_, b, kh, chunkStart, chunkLen);
                }
            }
            AscendC::SyncAll();
            return;
        }

        // 多核形态（双 Die NUMA 拓扑亲和硬绑定 + 全 AIV 核心分块网格饱和平铺）：
        // 解算本 AIV 所属物理 Die 与 Die 内局部 AIV 索引：
        int64_t physicalCore = (coreId_ >= p.cores) ? (coreId_ % p.cores) : coreId_;
        int64_t dieId = physicalCore % 2;
        int64_t coreInDie = physicalCore / 2;
        int64_t aivInDie = coreInDie * 2 + streamId_;
        int64_t aivsPerDie = p.cores; // 每 Die 拥有 (p.cores / 2) * 2 = p.cores 个 AIV

        int64_t pairCols = PairCols(p.hk);
        int64_t totalPairs = p.b * pairCols;

        // Pass 1: 均值预统计（严格归属本 Die 的 KV 头：kh = pCol * 2 + dieId；无条件执行，
        // smooth 关闭时仅落零值不读 GM——理由同单核分支注记）
        for (int64_t pairIdx = aivInDie; pairIdx < totalPairs; pairIdx += aivsPerDie) {
            int64_t b = pairIdx / pairCols;
            int64_t pCol = pairIdx % pairCols;
            int64_t kh = pCol * 2 + dieId;
            if (kh < p.hk) {
                kvPrep.ProcessHeadMeans(k_, v_, userWorkspace_, *tilingData_, b, kh);
            }
        }
        AscendC::SyncAll();

        // Pass 2: 128 行 Token 分块网格并行量化
        // 彻底根除 GQA 小头数场景（如 Hk=8 时仅 4 对）导致的 40 核空转长尾！
        // 总切片数 = totalPairs * chunksPerHead（如 16K 序列下 4 * 128 = 512 个分块）
        // 512 个分块任务在 Die 内全部 aivsPerDie 个 AIV 上彻底饱和分摊！
        int64_t totalDieTiles = totalPairs * chunksPerHead;
        for (int64_t tileIdx = aivInDie; tileIdx < totalDieTiles; tileIdx += aivsPerDie) {
            int64_t pairIdx = tileIdx / chunksPerHead;
            int64_t chunkIdx = tileIdx % chunksPerHead;
            int64_t b = pairIdx / pairCols;
            int64_t pCol = pairIdx % pairCols;
            int64_t kh = pCol * 2 + dieId;
            if (kh < p.hk) {
                int64_t chunkStart = chunkIdx * kKvStep;
                int64_t chunkLen = aicMin(kKvStep, p.kp - chunkStart);
                if (chunkLen > 0) {
                    kvPrep.ProcessHeadChunk(k_, v_, userWorkspace_, *tilingData_, b, kh, chunkStart, chunkLen);
                }
            }
        }
        AscendC::SyncAll();
    }

    __aicore__ inline Sa2Task DecodeTaskForCore(const Sa2Params &p, int64_t slot) const
    {
        if (p.cores <= 1) {
            int64_t die = (totalTasksPerDie_ > 0) ? (slot / totalTasksPerDie_) : 0;
            int64_t localGid = (totalTasksPerDie_ > 0) ? (slot % totalTasksPerDie_) : 0;
            return DecodeTask(p, die, localGid);
        }
        return DecodeTask(p, assignedDie_, slot);
    }

    /**
     * @brief AIC 负责：Cube MMAD 矩阵乘计算引擎，调度 INT8 QK^T 与 FP8 PV GEMM，两端交替调度 Stream 0 与 Stream 1
     */
    __aicore__ inline void ProcessAttentionCube(const Sa2Params &p)
    {
        if (range_.begin >= range_.end) {
            return;
        }

        constexpr Gran granMode = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
        Sa2AttentionLoop<Dtype, granMode, IsCausal> loop;
        loop.Init(q_, out_, lse_, userWorkspace_, *tilingData_);

        // P1-3 两深 lookahead（D-R13 三重协奏的真实现）：每流一条 Cube 游标状态——
        // issuedQk = 已发射的 QK 步数，drainedPv = 已排空的 PV 步数。先发 QK(j+1) 进 Cube
        // 管线（score ping-pong 双槽 + kQkFree 双信用），再阻塞等 AIV 的 P(j)（kPTaken）排空
        // PV(j)：AIV 做 softmax(j) 的窗口里 Cube 在跑 QK(j+1) 的 MMAD，消除旧实现
        // "QK(j)→PV(j)→QK(j+1) 全串行"的 AIC 空转。两流从区间两端对进、交替各推一步。
        struct StreamState {
            Sa2Task task{};
            bool has = false;
            int64_t issuedQk = 0;
            int64_t drainedPv = 0;
            int64_t slot = 0;       // 下一个候选槽（流 0 递增 / 流 1 递减）
        };
        const int64_t numTasks = range_.end - range_.begin;
        const int64_t mid = range_.begin + (numTasks + 1) / 2;

        StreamState streams[2];
        streams[0].slot = range_.begin;
        streams[1].slot = range_.end - 1;
        const int64_t slotStep[2] = {1, -1};

        // __aicore__ 显式标注：-xcce 混编下未标注 lambda 视作 __host__，不可由设备函数调用
        auto pullNext = [&](StreamState &st, int64_t stream) __aicore__ {
            // 流 0 正向扫 [begin, mid)；流 1 反向扫 [mid, end-1)（两端对进）
            while ((slotStep[stream] > 0) ? (st.slot < mid) : (st.slot >= mid)) {
                st.task = DecodeTaskForCore(p, st.slot);
                st.slot += slotStep[stream];
                if (st.task.valid && st.task.steps > 0) {
                    st.has = true;
                    st.issuedQk = 0;
                    st.drainedPv = 0;
                    return;
                }
            }
        };
        pullNext(streams[0], 0);
        pullNext(streams[1], 1);

        bool progress = true;
        while (streams[0].has || streams[1].has) {
            progress = false;
            for (int64_t stream = 0; stream < 2; ++stream) {
                StreamState &st = streams[stream];
                if (!st.has) {
                    continue;
                }
                // 1. QK 预发射：在途 ≤2（issuedQk - drainedPv < 2），先行打满两级深度
                while (st.issuedQk < st.task.steps && st.issuedQk - st.drainedPv < 2) {
                    loop.StepCubeQk(st.task, st.issuedQk, stream);
                    st.issuedQk++;
                    progress = true;
                }
                // 2. PV 排空：阻塞等 kPTaken（P 已由 AIV 落 L1）
                if (st.drainedPv < st.issuedQk) {
                    loop.StepCubePv(st.task, st.drainedPv, stream);
                    st.drainedPv++;
                    progress = true;
                }
                // 3. 任务收口：全步排空后归还双信用（FinishTileCube drain min(steps,2)）
                if (st.drainedPv >= st.task.steps) {
                    loop.FinishTileCube(st.task, stream);
                    st.has = false;
                    progress = true;
                    pullNext(st, stream);
                }
            }
            // 静态活性保障：任一 in-flight 任务必有 issuedQk > drainedPv 或可发射步，
            // 否则协议死锁（旗号对账守恒下不可达；真机压测在 M2b 出口判据复验）。
            if (!progress && (streams[0].has || streams[1].has)) {
                break;  // 防御性退出：不可达路径（宿主仿真与计数守恒证明下不触发）
            }
        }
    }

    /**
     * @brief AIV 负责：Vector 引擎，执行在线 Softmax（含 θ 阈值跳过）、FP8 P 离散化、O 累加与 LSE 计算
     */
    __aicore__ inline void ProcessAttentionVector(const Sa2Params &p)
    {
        if (range_.begin >= range_.end) {
            return;
        }

        constexpr Gran granMode = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
        Sa2AttentionLoop<Dtype, granMode, IsCausal> loop;
        loop.Init(q_, out_, lse_, userWorkspace_, *tilingData_);

        int64_t numTasks = range_.end - range_.begin;
        int64_t mid = range_.begin + (numTasks + 1) / 2;

        // DESIGN_SPEC §7：AIC 内两 AIV 流从区间两端对进
        if (streamId_ == 0) {
            // AIV0: 从前端正向推进 [range_.begin, mid)
            for (int64_t slot = range_.begin; slot < mid; ++slot) {
                Sa2Task task = DecodeTaskForCore(p, slot);
                if (!task.valid || task.steps <= 0) continue;
                loop.ProcessTileVector(task, 0);
            }
        } else {
            // AIV1: 从末端反向对进 [range_.end - 1, mid]
            for (int64_t slot = range_.end - 1; slot >= mid; --slot) {
                Sa2Task task = DecodeTaskForCore(p, slot);
                if (!task.valid || task.steps <= 0) continue;
                loop.ProcessTileVector(task, 1);
            }
        }
    }



    GM_ADDR q_{nullptr};
    GM_ADDR k_{nullptr};
    GM_ADDR v_{nullptr};
    GM_ADDR out_{nullptr};
    GM_ADDR lse_{nullptr};
    GM_ADDR userWorkspace_{nullptr};
    const Sa2TilingData *tilingData_{nullptr};
    int64_t coreId_{0};
    int64_t streamId_{0};
    int64_t assignedDie_{0};
    int64_t totalTasksPerDie_{0};
    Sa2Range range_{};
};

} // namespace sa2

// tiling 读取机制（M2b-0）：本 CANN 版本（9.2.0-beta.1 dav-c310）无 GET_TILING_DATA_WITH_STRUCT
// 宏（hoststub 假想 API）。真实机制 = 仓内 built-in 算子同款字段级标量搬运（__gm__ 指针逐字段
// 读入本地副本，参考 ops_legacy/get_padding_offset 的 InitTilingData / COPY_TILING_WITH_STRUCT）；
// 对 __gm__ 结构体做整体 deref 会依赖设备侧不可链接的 memcpy 调用，字段级读取为确定路径。
// Sa2TilingData 布局由 sa2_spec.h 的 sizeof/offsetof static_assert 冻结（256B），字段序禁止改动。
namespace sa2 {

__aicore__ inline void Sa2CopyTilingDataFromGm(const __gm__ Sa2TilingData *gm, Sa2TilingData *out)
{
    out->params.b = gm->params.b;
    out->params.hq = gm->params.hq;
    out->params.hk = gm->params.hk;
    out->params.sq = gm->params.sq;
    out->params.sk = gm->params.sk;
    out->params.d = gm->params.d;
    out->params.dp = gm->params.dp;
    out->params.qp = gm->params.qp;
    out->params.kp = gm->params.kp;
    out->params.qStride[0] = gm->params.qStride[0];
    out->params.qStride[1] = gm->params.qStride[1];
    out->params.qStride[2] = gm->params.qStride[2];
    out->params.kStride[0] = gm->params.kStride[0];
    out->params.kStride[1] = gm->params.kStride[1];
    out->params.kStride[2] = gm->params.kStride[2];
    out->params.vStride[0] = gm->params.vStride[0];
    out->params.vStride[1] = gm->params.vStride[1];
    out->params.vStride[2] = gm->params.vStride[2];
    out->params.causal = gm->params.causal;
    out->params.smoothK = gm->params.smoothK;
    out->params.smoothV = gm->params.smoothV;
    out->params.returnLse = gm->params.returnLse;
    out->params.gran = gm->params.gran;
    out->params.accum = gm->params.accum;
    out->params.smScale = gm->params.smScale;
    out->params.pDomain = gm->params.pDomain;
    out->params.lnPDomain = gm->params.lnPDomain;
    out->params.qFactor = gm->params.qFactor;
    out->params.kFactor = gm->params.kFactor;
    out->params.segWidth = gm->params.segWidth;
    out->params.cores = gm->params.cores;
    out->kMean = gm->kMean;
    out->vMean = gm->vMean;
    out->k8 = gm->k8;
    out->kScale = gm->kScale;
    out->v8 = gm->v8;
    out->vScale = gm->vScale;
}

} // namespace sa2

// 入口（M2b-0 重构为 col2im 同款形态）：非模板 __global__ + TILING_KEY_IS 运行期分发 +
// if constexpr 分支内隐式实例化模板 Kernel 类。迭代 2 修订：旧注记「显式实例化模板入口 +
// __simd_vf__ 调用链触发 AIV 后端 Unsupported scalar instruction」的根因结论已被控制变量
// 探针否证（判别条件 = AIV 编译单元内标量浮点 ALU/fcmp/浮点库调用与标量位强转，与入口
// 形态/调用链深度/显隐式实例化无关，证据链见 tests/aiv_backend_probe.cpp 与
// op_kernel/CMakeLists.txt 头注 R1/R2）；保留 col2im 形态是因为其工作，而非其曾归因的
// 原因。AIV pass 已于迭代 2 达成 EXIT=0（attention loop 三 VF 向量化改写）；merge 期
// undefined rintf = Q 量化/KV 预处理标量浮点核心（R1 剩余面，M2b Vector 通路整体向量化）。
// 8 组合的 key 值冻结于 sa2_template_key.h（FP16 = 1/513/257/769，BF16 = 27/539/283/795），
// 与 op_host tiling key 位序 [0:8) DTYPE、[8] GRAN、[9] CAUSAL 逐一对应。
__global__ __aicore__ void sa2_attention2(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(sa2::Sa2TilingData);
    const __gm__ sa2::Sa2TilingData *tilingGm = reinterpret_cast<__gm__ const sa2::Sa2TilingData *>(tiling);
    sa2::Sa2TilingData tilingData;
    sa2::Sa2CopyTilingDataFromGm(tilingGm, &tilingData);

    // tiling key 读一次进标量（TILING_KEY_IS 每次都是 workgroup_local GM 读，循环外单读
    // 避免逐分支重复 GM 读与内核体膨胀）。旧注记「后端报错实测定位为入口形态问题」已被
    // 控制变量探针否证：真实判别条件 = AIV 编译单元内标量浮点 ALU/fcmp/浮点库调用（R1）、
    // 「计算值/装载值 → float」标量位强转（R2，ISel Cannot select）、循环内标量参向量指令
    // 须循环不变（R3）——与入口形态/调用链深度/读次数无关；标量 tiling key 于循环外单读
    // 为合法形态（见本函数头注记与 tests/aiv_backend_probe.cpp）。
    const uint64_t tilingKey = TILING_KEY_VAR;
    if (tilingKey == 1) {          // FP16 + per_warp + 非因果
        sa2::SageAttention2Kernel<half, false, false> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 513) { // FP16 + per_warp + causal
        sa2::SageAttention2Kernel<half, false, true> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 257) { // FP16 + per_thread + 非因果
        sa2::SageAttention2Kernel<half, true, false> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 769) { // FP16 + per_thread + causal
        sa2::SageAttention2Kernel<half, true, true> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 27) {  // BF16 + per_warp + 非因果
        sa2::SageAttention2Kernel<bfloat16_t, false, false> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 539) { // BF16 + per_warp + causal
        sa2::SageAttention2Kernel<bfloat16_t, false, true> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 283) { // BF16 + per_thread + 非因果
        sa2::SageAttention2Kernel<bfloat16_t, true, false> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    } else if (tilingKey == 795) { // BF16 + per_thread + causal
        sa2::SageAttention2Kernel<bfloat16_t, true, true> kernel;
        kernel.Init(q, k, v, out, lse, workspace, &tilingData);
        kernel.Process();
    }
}
