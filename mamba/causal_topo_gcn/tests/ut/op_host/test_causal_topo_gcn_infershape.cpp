#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CausalTopoGcnInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(CausalTopoGcnInfershapeTest, test_case_basic) {
    gert::InfershapeContextPara infershapeContextPara(
        "CausalTopoGCN",
        {
            {{{1, 9, 2048}, {1, 9, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 9, 2048}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
