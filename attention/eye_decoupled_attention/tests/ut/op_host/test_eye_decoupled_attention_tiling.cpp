#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class EyeDecoupledAttentionTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(EyeDecoupledAttentionTilingTest, test_case_basic) {
    gert::TilingContextPara tilingContextPara(
        "EyeDecoupledAttention",
        {
            {{{2, 4096, 128}, {2, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 128}, {1, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{2, 4096, 128}, {2, 4096, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            gert::TilingContextPara::OpAttr("C", AnyValue::CreateFrom<int64_t>(128)),
            gert::TilingContextPara::OpAttr("hb", AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("wb", AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("bn_h", AnyValue::CreateFrom<int64_t>(8)),
            gert::TilingContextPara::OpAttr("bn_w", AnyValue::CreateFrom<int64_t>(8))
        },
        nullptr, 64, 262144, 4096
    );
    uint64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
