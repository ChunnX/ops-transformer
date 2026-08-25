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

namespace {
bool RunScenario(ge::Session &session, uint32_t graphId, const std::string &scenario,
                 const DeclaredShapes &declared, std::vector<CaseSpec> cases)
{
    ge::Graph graph((scenario + "_graph").c_str());
    BuildGraph(scenario, cases[0], declared, graph);
    std::map<ge::AscendString, ge::AscendString> graphOptions;
    const auto addStatus = session.AddGraph(graphId, graph, graphOptions);
    if (addStatus != ge::SUCCESS) {
        std::cerr << "ERROR: AddGraph status=" << addStatus << std::endl;
        return false;
    }

    std::cout << "Scenario " << scenario << ", declared shape "
              << ShapeBundleJson(declared.tokens, declared.indices, declared.probs) << std::endl;
    size_t passed = 0U;
    for (auto &spec : cases) {
        const auto concrete = ShapeBundleJson(TokensShape(spec), IndicesShape(spec), ProbsShape(spec));
        std::cout << "Run concrete shape " << concrete << std::endl;
        double durationUs = 0.0;
        passed += RunAndVerify(session, graphId, spec, durationUs) ? 1U : 0U;
    }
    std::cout << "Scenario " << scenario << " summary: " << passed << "/" << cases.size() << " passed"
              << std::endl;
    return passed == cases.size();
}
}  // namespace

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
    std::vector<CaseSpec> concreteCases = {
        {"dynamic_2x4", 2, 4, 2, {1, 3}, true, ge::DT_FLOAT, ge::DT_FLOAT},
        {"dynamic_4x7", 4, 7, 2, {1, 3}, true, ge::DT_FLOAT, ge::DT_FLOAT},
        {"dynamic_8x16", 8, 16, 2, {1, 3}, true, ge::DT_FLOAT, ge::DT_FLOAT},
    };

    const bool dimPassed = RunScenario(
        *session, 201U, "unknown_dim_minus_1", {{-1, -1}, {-1}, {-1, -1}}, concreteCases);
    const bool rankPassed = RunScenario(
        *session, 202U, "unknown_rank_minus_2", {{-2}, {-2}, {-2}}, concreteCases);
    delete session;
    ge::GEFinalize();
    if (!dimPassed || !rankPassed) {
        std::cerr << "ERROR: dynamic GEIR descriptor execution did not pass" << std::endl;
        return 1;
    }
    std::cout << "MoeTokenUnpermuteWithEp dynamic GEIR descriptor execution PASSED "
                 "(-1 and -2; concrete protocol ranks remain 2/1/2)" << std::endl;
    return 0;
}
