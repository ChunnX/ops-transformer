/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CV_TASK_SCHEDULER_H_
#define CV_TASK_SCHEDULER_H_

namespace AttentionTask {
template <unsigned Preload, unsigned Capacity, typename Task>
struct CvTaskScheduler {
    static_assert(Capacity >= Preload + 1 && (Capacity & (Capacity - 1)) == 0,
                  "Task ring must be a power of two and hold the pipeline");
    template <typename Hooks, typename Block>
    __aicore__ static inline void Run(Hooks &hooks, Block &block)
    {
        Task tasks[Capacity] = {};
        unsigned created = 0;
        unsigned executed = 0;
        unsigned pending = 0;
        bool more = true;
        while (more || pending) {
            more = hooks.NextTask(created, tasks[created & (Capacity - 1)]);
            if (more) {
                ++created;
                ++pending;
            }
            if (pending) {
                Task &current = tasks[executed & (Capacity - 1)];
                if (current.isValid) {
                    block.ComputeFirst(current);
                }
                if (executed >= Preload) {
                    Task &previous = tasks[(executed - Preload) & (Capacity - 1)];
                    if (previous.isValid) {
                        block.ComputeSecond(previous);
                        previous.isValid = false;
                    }
                }
                ++executed;
                if (executed > Preload) {
                    --pending;
                }
            }
        }
    }
};
} // namespace AttentionTask
#endif
