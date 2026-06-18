#ifndef FLASH_ATTN_TASK_LOOP_HOOKS_H_
#define FLASH_ATTN_TASK_LOOP_HOOKS_H_

#include "../../../common/op_kernel/const_def.h"

namespace BaseApi {

template<typename Kernel>
__aicore__ inline void CommonTaskLoop(Kernel &self, uint32_t sectionIdx)
{
    using Hooks = typename Kernel::TaskLoopHooks;

    Hooks::InitLoopState(self);
    Hooks::GetSectionInfo(self, sectionIdx);

    typename Kernel::RunInfoType taskRunInfo[Kernel::PRELOAD_TASK_CACHE_SIZE] = {};

    self.createdTaskCount = 0;
    Hooks::ResetLoopState(self);

    uint32_t bN2Cur = self.bN2Start_;
    uint32_t gS1Cur = self.gS1OStart_;
    uint32_t s2Cur = self.s2OStart_;
    self.prevBN2Idx = bN2Cur;
    self.prevGS1Idx = gS1Cur;

    uint32_t executedTaskCount = 0;
    bool shouldDispatchTask = true;
    uint32_t validTaskCount = 0;
    while (shouldDispatchTask || validTaskCount) {
        shouldDispatchTask = self.ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = self.GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);

            if constexpr (Hooks::HAS_DEAL_ZERO) {
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    Hooks::CreateTask(self, taskRunInfo, self.createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    self.createdTaskCount++;
                    validTaskCount++;
                    Hooks::UpdateAxisInfo(self, taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    Hooks::DealZeroActSeqLen(self, bN2Cur);
                    Hooks::UpdateAxisInfo(self, taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    Hooks::UpdateAxisInfo(self, taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            } else {
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    Hooks::CreateTask(self, taskRunInfo, self.createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    self.createdTaskCount++;
                    validTaskCount++;
                }
                Hooks::UpdateAxisInfo(self, taskDealMode, bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode != TASK_DEAL_MODE::CREATE_TASK) {
                    continue;
                }
            }
        }
        if (validTaskCount) {
            Hooks::ExecuteTask(self, taskRunInfo, executedTaskCount);
            executedTaskCount++;
            if (executedTaskCount > Kernel::PRELOAD_N) {
                validTaskCount--;
            }
        }
    }
}

} // namespace BaseApi

#endif // FLASH_ATTN_TASK_LOOP_HOOKS_H_
