/**
 * mgb_kts_switch.h — KTS 总开关（唯一改动点）
 * 生产/合入构建：MGB_KTS_ENABLE 0（所有打点宏展开为空，二进制零残留）
 * 诊断构建：由 tools/kts_build.sh 把下行切为 1（该轮 y 输出被 trace 占用，禁精度比对）
 */
#ifndef MGB_KTS_SWITCH_H
#define MGB_KTS_SWITCH_H
#define MGB_KTS_ENABLE 0
#endif
