// Included after the runner extracts the exact production CCL bound function.
// Compare its integer allocation against production WorkspaceLayout; this is
// not a second implementation of either allocation formula.
static int RunTorchBoundChecks()
{
    const uint32_t worlds[] = {2, 4, 8, 32, 128, 1024};
    const uint32_t experts[] = {1, 4, 32, 256};
    const uint32_t tokenCaps[] = {1, 128, 8192};
    const uint32_t hidden[] = {1024, 4096, 8192};
    const uint32_t intermediate[] = {256, 2048, 4096};
    const uint32_t topks[] = {1, 8, 32};
    const uint32_t maxReceive[] = {0, 1, 128};
    const uint32_t cores[] = {1, 28, 36};
    for (uint32_t world : worlds) {
        for (uint32_t perRank : experts) {
            if (world * perRank > 2048) {
                continue;
            }
            for (uint32_t tokenCap : tokenCaps) {
                for (uint32_t h : hidden) {
                    for (uint32_t i : intermediate) {
                        for (uint32_t topk : topks) {
                            for (uint32_t maxRecv : maxReceive) {
                                for (uint32_t topology = 0; topology < 2; ++topology) {
                                    for (uint32_t prefetch = 0; prefetch < 2; ++prefetch) {
                                        const int64_t fullBound = TorchBound::CalcCclWorkspaceSizeA5(
                                            world, world * perRank, tokenCap, topk, h, i, maxRecv, prefetch,
                                            topology == 0 ? 0 : 2, false);
                                        const int64_t ringBound = topology == 0 ? TorchBound::CalcCclWorkspaceSizeA5(
                                            world, world * perRank, tokenCap, topk, h, i, maxRecv, prefetch, 0, true) : 0;
                                        for (uint32_t core : cores) {
                                            for (uint8_t mode = 0; mode <= GMM_MODE_A4W4_NZ; ++mode) {
                                                for (int32_t dedup = 0; dedup < (topology == 0 ? 4 : 1); ++dedup) {
                                                    for (uint32_t quant = 0; quant < 2; ++quant) {
                                                        for (uint32_t shared = 0; shared <= 4; shared += 4) {
                                                            MegaMoeTilingData tiling{};
                                                            tiling.h = h;
                                                            tiling.hiddenDim = i * 2;
                                                            tiling.epWorldSize = world;
                                                            tiling.rankNumPerServer = world / 2;
                                                            tiling.numMaxTokensPerRank = tokenCap;
                                                            tiling.bs = tokenCap;
                                                            tiling.maxOutputSize = maxRecv != 0 ? maxRecv :
                                                                tokenCap * world * (topk < perRank ? topk : perRank);
                                                            tiling.topK = topk;
                                                            tiling.aicNum = core;
                                                            tiling.moeExpertPerRank = perRank;
                                                            tiling.topoType = topology;
                                                            tiling.moeGmmMode = mode;
                                                            tiling.sharedGmmMode = mode;
                                                            tiling.dedupMode = dedup;
                                                            tiling.combineQuantMode = quant == 0 ? 0 : 3;
                                                            tiling.topkWeightsPrefetch = prefetch;
                                                            tiling.sharedExpertNum = shared;
                                                            tiling.isSharedQuantIndependent = true;
                                                            const uint32_t n1 = (tiling.hiddenDim + 255) / 256;
                                                            const uint32_t n2 = (h + 255) / 256;
                                                            const uint32_t w1 = (core * 4 + n1 - 1) / n1;
                                                            const uint32_t w2 = (core + n2 - 1) / n2;
                                                            tiling.mGroupsPerWave = w1 > w2 ? w1 : w2;
                                                            const uint32_t tiles = ((tiling.maxOutputSize + 255) / 256) * n1;
                                                            tiling.maxTilesPerExpert = ((tiles + 15) / 16) * 16;
                                                            const uint64_t tokenGroups = (static_cast<uint64_t>(tokenCap) *
                                                                                          world + 255) / 256;
                                                            tiling.combineSyncSlotCountPerExpert = tokenGroups > core ?
                                                                                                      tokenGroups : core;
                                                            const WorkspaceLayout layout(&tiling);
                                                            const bool ring = IsMteRingBufferEnabled(&tiling);
                                                            const int64_t bound = ring ? ringBound : fullBound;
                                                            if (layout.workspaceSize > bound) {
                                                                return ring ? 41 : 42;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
