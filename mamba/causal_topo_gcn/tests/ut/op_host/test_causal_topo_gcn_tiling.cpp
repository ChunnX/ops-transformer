#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class CausalTopoGcnTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(CausalTopoGcnTilingTest, test_case_basic) {
    gert::TilingContextPara tilingContextPara(
        "CausalTopoGCN",
        {
            {{{1, 9, 2048}, {1, 9, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{1, 9, 2048}, {1, 9, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            gert::TilingContextPara::OpAttr("in_channels", AnyValue::CreateFrom<int64_t>(2048)),
            gert::TilingContextPara::OpAttr("out_channels", AnyValue::CreateFrom<int64_t>(2048))
        },
        nullptr, 64, 262144, 4096
    );
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
