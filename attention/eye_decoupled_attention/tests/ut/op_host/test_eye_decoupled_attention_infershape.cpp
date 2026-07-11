#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class EyeDecoupledAttentionInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(EyeDecoupledAttentionInfershapeTest, test_case_basic) {
    gert::InfershapeContextPara infershapeContextPara(
        "EyeDecoupledAttention",
        {
            {{{2, 4096, 128}, {2, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4096, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
