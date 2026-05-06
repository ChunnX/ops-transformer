/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel_run_context_faker.h"
#include "mega_moe.cpp"
#include "mega_moe_tiling_def.h"

using namespace std;

class MegaMoeKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MegaMoeKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MegaMoeKernelTest TearDown" << std::endl;
    }
};

TEST_F(MegaMoeKernelTest, ascend950_bf16_h4096_n1024) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_bf16_h5120_n2048) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 256;
    tilingDef.h = 5120;
    tilingDef.topk = 8;
    tilingDef.expertPerRank = 8;
    tilingDef.n = 2048;
    tilingDef.epWorldSize = 8;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 8, 8, 8, 8, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_bf16_h7168_n7168) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 512;
    tilingDef.h = 7168;
    tilingDef.topk = 8;
    tilingDef.expertPerRank = 2;
    tilingDef.n = 7168;
    tilingDef.epWorldSize = 2;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 2, 2, 2, 2, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_fp8_e5m2) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 23;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_max_bs) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 512;
    tilingDef.h = 4096;
    tilingDef.topk = 8;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_min_bs) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 1;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_min_expert_per_rank) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 1;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 16;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_max_expert_per_rank) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 8;
    tilingDef.expertPerRank = 16;
    tilingDef.n = 1024;
    tilingDef.epWorldSize = 2;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 16, 16, 16, 16, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_different_n_values) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 2048;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}

TEST_F(MegaMoeKernelTest, ascend950_different_n_values_3072) {
    MegaMoeTilingDef tilingDef;
    tilingDef.bs = 128;
    tilingDef.h = 4096;
    tilingDef.topk = 6;
    tilingDef.expertPerRank = 4;
    tilingDef.n = 3072;
    tilingDef.epWorldSize = 4;
    tilingDef.dispatchQuantOutType = 24;
    tilingDef.coreNum = 64;
    tilingDef.ubSize = 253952;
    
    InitMegaMoeTiling(tilingDef);
    
    auto holder = gert::KernelRunContextFaker()
                     .NodeIoNum(10, 2)
                     .IrInstanceNum({1, 1, 1, 1, 4, 4, 4, 4, 1, 1})
                     .KernelName("MegaMoe")
                     .Build();
    
    EXPECT_EQ(holder.GetContext<gert::KernelContext>()->GetNodeType(), "MegaMoe");
}