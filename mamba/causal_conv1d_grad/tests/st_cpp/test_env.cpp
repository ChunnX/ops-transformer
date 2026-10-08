#include <cstdlib>
#include <cstdio>
extern "C" {
    void __attribute__((constructor)) set_opp_path_before_main() {
        const char* custom_opp = "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/ops-registry-invoke/operators/causal_conv1d_grad/custom_opp";
        const char* ascend_home = getenv("ASCEND_HOME_PATH");
        if (ascend_home) {
            char opp_path[2048];
            snprintf(opp_path, sizeof(opp_path), "%s:%s/opp", custom_opp, ascend_home);
            setenv("ASCEND_OPP_PATH", opp_path, 1);
            printf("[CONSTRUCTOR] Set ASCEND_OPP_PATH=%s\n", opp_path);
        } else {
            printf("[CONSTRUCTOR] ASCEND_HOME_PATH not set!\n");
        }
    }
}

int main() {
    printf("main: ASCEND_OPP_PATH=%s\n", getenv("ASCEND_OPP_PATH"));
    return 0;
}
