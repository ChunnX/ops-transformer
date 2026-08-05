/**
 * \file profiler_config.h
 * \brief Chip-specific profiler constants — static version for ops 仓子包路径.
 *
 * 子包路径（build.sh --pkg）没有 CMake configure_file 基础设施，因此使用静态 header。
 * 数值来自 host_check_cores / parse_prof.py KNOWN_CONFIGS（Ascend 950 MIX_AIC_1_2:
 * 56 AIV + 28 AIC = 84 cores）。其他芯片请运行 host_check_cores 获取正确值后替换。
 *
 * ⚠️ 必须与 mega_moe kernel 的 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) 匹配：
 *   - AIV 核 GetBlockIdx() = 0..55   → ring slot 0..55
 *   - AIC 核 GetBlockIdx() = 0..27   → ring slot = GetBlockIdx() + PROF_AIV_CORES = 56..83
 */

#pragma once

// Total core count (AIV + AIC for MIX mode, AIV only for AIV-only mode).
// [MODIFY] 运行 host_check_cores 获取实际值后替换。
#ifndef PROF_N_CORES
#define PROF_N_CORES 84
#endif

// AIV core count. Used as AIC offset in GetCoreId().
// [MODIFY] 运行 host_check_cores 获取实际值后替换。
#ifndef PROF_AIV_CORES
#define PROF_AIV_CORES 56
#endif

// Buffer layout constants (shared between device and host)
// Max events per core. Default 4096 is sufficient for ~256 tiles at 16 events/tile.
#ifndef PROF_SLOTS_PER_CORE
#define PROF_SLOTS_PER_CORE 4096
#endif
#define PROF_RECORD_BYTES  16
#define PROF_RINGIDX_ALIGN 64

// Derived buffer layout constants
#define PROF_RINGIDX_REGION  (PROF_N_CORES * PROF_RINGIDX_ALIGN)
#define PROF_DATA_REGION     (PROF_N_CORES * PROF_SLOTS_PER_CORE * PROF_RECORD_BYTES)
#define PROF_BUF_TOTAL       (PROF_RINGIDX_REGION + PROF_DATA_REGION)

// Overflow sentinel value — written to the last slot's event_id field when the
// ring buffer overflows. IMPORTANT: User event IDs in profiler.h MUST NOT use 0xFFFF.
#define EVT_OVERFLOW 0xFFFF

// Profiler version — single source of truth for both device and host headers.
#define PROFILER_VERSION "1.0.0"
