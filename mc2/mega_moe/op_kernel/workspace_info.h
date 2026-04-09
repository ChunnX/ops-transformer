#ifndef WORKSPACE_INFO_H
#define WORKSPACE_INFO_H
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#endif
#include "mega_moe_tiling_def.h"
#include "gmmsq_base.h"
enum TaskType {
    TASK_TYPE_0 = 0,
    TASK_TYPE_1 = 1,
    TASK_TYPE_2 = 2,
    TASK_TYPE_COUNT = 3
};

// 每个task的最大group数量
#define MAX_GROUPS_PER_TASK 16
// 队列容量
#define QUEUE_CAPACITY 1024

// Tile OP数据结构
struct TileOp {
    uint32_t task_type;   // 任务类型（0, 1, 2）    0 alltoall 1 gmm1 2 gmm2
    uint32_t group_id;    // Group ID
    uint32_t tile_id;     // Tile ID，最小调度单位
};

// 任务队列结构
struct TaskQueue {
    TileOp data[QUEUE_CAPACITY];  // 队列数据数组
    uint32_t head;        // 队列头（原子操作）
    uint32_t tail;        // 队列尾
    uint32_t capacity;    // 队列容量
};

// 每个group的状态信息
struct GroupStatus {
    uint32_t total_tiles;     // 该group的总tile数
    uint32_t completed_tiles; // 已完成的tile数（原子操作）
    bool is_completed;        // 该group是否已完成
};

// 每个task的信息
struct TaskInfo {
    GroupStatus groups[MAX_GROUPS_PER_TASK];   // 该task下所有group的状态
    uint32_t group_count;  // group数量
};

// 调度器上下文
struct SchedulerContext {
    TaskQueue queue;               // 任务队列
    TaskInfo tasks[TASK_TYPE_COUNT]; // 所有task的信息
    uint32_t global_completion;   // 全局完成标志（原子操作）
    uint32_t total_groups;         // 总group数
};




struct WorkspaceInfo{
    GM_ADDR gmCtx;
    GM_ADDR ptrA;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrFlagList;
    GM_ADDR ptrFlagList2;
    int64_t workspaceSize;
    CATLASS_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData) {
        workspaceSize = 0;
        ptrA = base;
        workspaceSize += CeilAlign(tilingData->maxOutputSize * tilingData->k * sizeof(int8_t), ALIGN_512);
        ptrAScale = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->maxOutputSize * tilingData->k / MXFP_SCALE_GROUP_NUM * sizeof(int8_t), ALIGN_512);
        ptrA2 = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->maxOutputSize * tilingData->n / SWIGLU_N_HALF * sizeof(int8_t), ALIGN_512);
        ptrA2Scale = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->maxOutputSize * tilingData->n / SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM * sizeof(int8_t), ALIGN_512);
        ptrcumsumMM = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->EP * tilingData->expertPerRank * sizeof(int32_t), ALIGN_512);
        expandedRowIdx = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->m, 256) * tilingData->topK * sizeof(int32_t);
        ptrSumBeforeRank = base + workspaceSize;
        workspaceSize += CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128) * sizeof(int32_t);
        ptrFlagList = base + workspaceSize;
        workspaceSize += tilingData->expertPerRank * INT_CACHELINE * sizeof(int32_t);
        ptrFlagList2 = base + workspaceSize;
        workspaceSize += tilingData->expertPerRank * INT_CACHELINE * sizeof(int32_t);
        gmCtx = base + workspaceSize;
        workspaceSize += CeilAlign(sizeof(SchedulerContext), ALIGN_512);
        expandedRowIdxGather = ptrA;    // 复用
    }
};

struct PeermemInfo {
    GM_ADDR base;
    GM_ADDR ptrA0;      // initrouting结果，包括data+scale
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrD;
    CATLASS_DEVICE PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData) {
        int64_t offset = PEERMEM_DATA_OFFSET;
        
        ptrA0 = base + offset;
        offset += tilingData->m * tilingData->topK * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
        ptrTokenPerExpert = base + offset;
        offset += tilingData->EP * CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128) * sizeof(int32_t);
        ptrD = base + offset;
    }
};

class Layout3D {
    int64_t strides[2];
    public:
    CATLASS_DEVICE
    Layout3D() {}
    CATLASS_DEVICE
    Layout3D(int64_t stride0, int64_t stride1) {
        strides[0] = stride0;
        strides[1] = stride1;
    }
    CATLASS_DEVICE
    int64_t operator() (int64_t dim0, int64_t dim1, int64_t dim2) {
        return dim0 * strides[0] + dim1 * strides[1] + dim2;
    }
};
#endif