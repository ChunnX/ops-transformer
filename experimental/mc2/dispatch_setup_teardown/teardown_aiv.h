#ifndef TEARDOWN_AIV_H
#define TEARDOWN_AIV_H
#include <cstdint>
// dispatch_teardown(AIV <<<>>>): 忙等轮询状态区, 见非0往下走并写 done
void dispatch_teardown_demo(uint32_t blockDim, void* stream, uint8_t* statusGM, uint8_t* doneGM);
#endif  // TEARDOWN_AIV_H
