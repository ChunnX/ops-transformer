/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_unpermute_with_ep_geir_test_common.h"

using namespace moe_token_unpermute_with_ep_geir_test;

int main()
{
    if (!InitializeGe()) {
        return 1;
    }
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    auto *session = new (std::nothrow) ge::Session(sessionOptions);
    if (session == nullptr) {
        ge::GEFinalize();
        return 1;
    }
    const std::vector<ge::DataType> floatDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
    std::vector<CaseSpec> cases;
    for (const auto tokenDtype : floatDtypes) {
        for (const auto probsDtype : floatDtypes) {
            cases.push_back({"static_dtype_combo", 2, 4, 2, {1, 3}, true, tokenDtype, probsDtype});
        }
        cases.push_back({"static_no_probs", 2, 4, 2, {1, 3}, false, tokenDtype, tokenDtype});
    }
    cases.push_back({"static_fallback_probs", 3, 5, 2, {0, 0}, true, ge::DT_FLOAT, ge::DT_BF16});
    cases.push_back({"static_fallback_no_probs", 5, 7, 1, {0, 0}, false, ge::DT_BF16, ge::DT_BF16});

    uint32_t graphId = 100U;
    size_t passed = 0U;
    for (size_t index = 0; index < cases.size(); ++index, ++graphId) {
        cases[index].name += "_" + std::to_string(index);
        ge::Graph graph(("static_graph_" + std::to_string(index)).c_str());
        BuildGraph("static_" + std::to_string(index), cases[index], ConcreteDeclaredShapes(cases[index]), graph);
        std::map<ge::AscendString, ge::AscendString> graphOptions;
        const auto addStatus = session->AddGraph(graphId, graph, graphOptions);
        if (addStatus != ge::SUCCESS) {
            std::cerr << "ERROR: AddGraph status=" << addStatus << std::endl;
            delete session;
            ge::GEFinalize();
            return 1;
        }
        double durationUs = 0.0;
        passed += RunAndVerify(*session, graphId, cases[index], durationUs) ? 1U : 0U;
    }
    delete session;
    ge::GEFinalize();
    if (passed != cases.size()) {
        std::cerr << "ERROR: static summary " << passed << "/" << cases.size() << std::endl;
        return 1;
    }
    std::cout << "MoeTokenUnpermuteWithEp static GEIR verification PASSED" << std::endl;
    return 0;
}
