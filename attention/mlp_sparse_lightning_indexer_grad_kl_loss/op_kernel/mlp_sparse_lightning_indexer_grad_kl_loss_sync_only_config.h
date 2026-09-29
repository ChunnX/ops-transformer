/*
 * Compile-time controls for the deterministic synchronization-only kernel.
 */
#ifndef MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SYNC_ONLY_CONFIG_H
#define MLP_SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SYNC_ONLY_CONFIG_H

#ifndef SLI_SYNC_ONLY_VV_EVENT_ID
// Match before by default; use 15 only for an isolated-ID A/B control.
#define SLI_SYNC_ONLY_VV_EVENT_ID 8
#endif

#ifndef SLI_SYNC_ONLY_ISOLATE_V2_EVENT
// V2 mode-0 flags: 0 = {0,1}; 1 = {15,15}; 2 = {15,1}; 3 = {0,15}.
// Modes 2/3 isolate one parity while preserving the two-ID structure.
#define SLI_SYNC_ONLY_ISOLATE_V2_EVENT 0
#endif

#ifndef SLI_SYNC_ONLY_DELAY_ENABLE
#define SLI_SYNC_ONLY_DELAY_ENABLE 0
#endif
#ifndef SLI_SYNC_ONLY_DELAY_MODE
#define SLI_SYNC_ONLY_DELAY_MODE 2
#endif
#ifndef SLI_SYNC_ONLY_DELAY_COUNT
#define SLI_SYNC_ONLY_DELAY_COUNT 1000
#endif
#ifndef SLI_SYNC_ONLY_RANDOM_SEED
#define SLI_SYNC_ONLY_RANDOM_SEED 0x13579BDFU
#endif
#ifndef SLI_SYNC_ONLY_DELAY_TARGET_AIC
#define SLI_SYNC_ONLY_DELAY_TARGET_AIC 0
#endif
#ifndef SLI_SYNC_ONLY_DELAY_TARGET_AIV
#define SLI_SYNC_ONLY_DELAY_TARGET_AIV -1
#endif
#ifndef SLI_SYNC_ONLY_DELAY_FIRST_TASK
#define SLI_SYNC_ONLY_DELAY_FIRST_TASK 0
#endif
#ifndef SLI_SYNC_ONLY_DELAY_TASK_STRIDE
#define SLI_SYNC_ONLY_DELAY_TASK_STRIDE 2
#endif
#ifndef SLI_SYNC_ONLY_DELAY_MAX_HITS
#define SLI_SYNC_ONLY_DELAY_MAX_HITS 0
#endif

#if SLI_SYNC_ONLY_VV_EVENT_ID != 8 && SLI_SYNC_ONLY_VV_EVENT_ID != 15
#error "VV eventId must be 8 (before) or 15 (isolated A/B control); 6 conflicts with C1->V1 SY"
#endif
#if SLI_SYNC_ONLY_ISOLATE_V2_EVENT < 0 || SLI_SYNC_ONLY_ISOLATE_V2_EVENT > 3
#error "SLI_SYNC_ONLY_ISOLATE_V2_EVENT must be 0, 1, 2, or 3"
#endif
#if SLI_SYNC_ONLY_ISOLATE_V2_EVENT && SLI_SYNC_ONLY_VV_EVENT_ID == 15
#error "VV and V2 cannot both use isolated event 15"
#endif
#if SLI_SYNC_ONLY_DELAY_ENABLE != 0 && SLI_SYNC_ONLY_DELAY_ENABLE != 1
#error "SLI_SYNC_ONLY_DELAY_ENABLE must be 0 or 1"
#endif
#if SLI_SYNC_ONLY_DELAY_MODE < 1 || SLI_SYNC_ONLY_DELAY_MODE > 2 || SLI_SYNC_ONLY_DELAY_COUNT < 0
#error "invalid delay mode/count"
#endif
#if SLI_SYNC_ONLY_DELAY_TARGET_AIC < -1 || SLI_SYNC_ONLY_DELAY_TARGET_AIV < -1 || \
    SLI_SYNC_ONLY_DELAY_TARGET_AIV > 1
#error "invalid delay target"
#endif
#if SLI_SYNC_ONLY_DELAY_FIRST_TASK < 0 || SLI_SYNC_ONLY_DELAY_TASK_STRIDE <= 0 || \
    SLI_SYNC_ONLY_DELAY_MAX_HITS < 0
#error "invalid delay task selection"
#endif

#endif
