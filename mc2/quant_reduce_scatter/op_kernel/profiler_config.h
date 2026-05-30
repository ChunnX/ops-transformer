/**
 * \file profiler_config.h
 * \brief Chip-specific profiler constants for Ascend 950PR (AIV-only kernel).
 *
 * Static version — no CMake configure_file needed.
 * Both device-side profiler_core.h and host-side code include this header.
 *
 * Ascend 950PR AIV-only: 56 AIV cores, 0 AIC cores.
 */

#pragma once

#ifndef PROF_N_CORES
#define PROF_N_CORES 56
#endif

#ifndef PROF_AIV_CORES
#define PROF_AIV_CORES 0
#endif

// Buffer layout constants (shared between device and host)
#define PROF_SLOTS_PER_CORE 4096
#define PROF_RECORD_BYTES  16
#define PROF_RINGIDX_ALIGN 64
