// sa2_sim_engine.h — SageAttention2 宿主仿真算子引擎（纯标准 C++17，零硬件依赖）
#ifndef SA2_SIM_ENGINE_H_
#define SA2_SIM_ENGINE_H_

#include <cstdint>
#include <vector>

#include "sa2_spec.h"
#include "sa2_planner.h"
#include "sa2_sim_kv_prep.h"
#include "sa2_sim_loop.h"

namespace sa2 {
namespace sim {

template <typename Dtype, bool GranPerThread, bool IsCausal>
class Sa2SimulationEngine {
public:
    inline Sa2SimulationEngine() = default;

    inline void Init(
        uint8_t *q, uint8_t *k, uint8_t *v,
        uint8_t *out, uint8_t *lse, uint8_t *workspace,
        const Sa2TilingData *tilingData)
    {
        q_ = q;
        k_ = k;
        v_ = v;
        out_ = out;
        lse_ = lse;
        userWorkspace_ = workspace;
        tilingData_ = tilingData;
    }

    inline void Process()
    {
        ProcessGolden();
    }

    inline void ProcessGolden()
    {
        const Sa2Params &p = tilingData_->params;

        // 1. 全量 KV 预处理（单线程顺序）
        {
            constexpr Gran granMode = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
            Sa2KvPreprocessorSim<Dtype, granMode> kvPrep;
            for (int64_t kvIdx = 0; kvIdx < p.b * p.hk; ++kvIdx) {
                kvPrep.ProcessHead(k_, v_, userWorkspace_, *tilingData_, kvIdx / p.hk, kvIdx % p.hk);
            }
        }

        // 2. 双 Die 全任务（单线程复用）
        constexpr Gran granMode = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
        Sa2AttentionLoopSim<Dtype, granMode, IsCausal> loop;
        loop.Init(q_, out_, lse_, userWorkspace_, *tilingData_);

        for (int64_t die = 0; die < 2; ++die) {
            const int64_t pairCols = PairCols(p.hk);
            const int64_t totalTasksPerDie = TasksPerDie(SlotsPerDie(p.b, pairCols),
                                                          PerHead(GroupSize(p.hq, p.hk), TileCount(p.qp)));
            for (int64_t gid = 0; gid < totalTasksPerDie; ++gid) {
                Sa2Task task = DecodeTask(p, die, gid);
                if (!task.valid || task.steps <= 0) continue;
                loop.ProcessTile(task, 0);
            }
        }
    }

private:
    uint8_t *q_{nullptr};
    uint8_t *k_{nullptr};
    uint8_t *v_{nullptr};
    uint8_t *out_{nullptr};
    uint8_t *lse_{nullptr};
    uint8_t *userWorkspace_{nullptr};
    const Sa2TilingData *tilingData_{nullptr};
};

} // namespace sim
} // namespace sa2

#endif // SA2_SIM_ENGINE_H_
