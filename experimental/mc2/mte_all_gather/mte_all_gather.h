#ifndef MTE_ALL_GATHER_H
#define MTE_ALL_GATHER_H

void mte_all_gather_demo(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* output,
                           uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);

#endif //MTE_ALL_GATHER_H
