std::vector<int64_t> LightningOutputShape(const Args &args)
{
    if (args.layoutQuery == "TND") {
        return {args.queryShape[0], 1, args.sparseCount};
    }
    return {args.queryShape[0], args.queryShape[1], 1, args.sparseCount};
}

struct LightningIndexerGolden {
    std::vector<double> indices;
    std::vector<double> values;
};

struct RecurrentGatedDeltaRuleGolden {
    std::vector<double> out;
    std::vector<double> state;
};

int64_t LightningQueryBatchSize(const Args &args)
{
    if (args.layoutQuery == "TND") {
        return static_cast<int64_t>(args.actualSeqLengthsQuery.size());
    }
    return args.queryShape[0];
}

int64_t LightningActualSeqLen(
    const std::vector<int32_t> &seqLengths, int64_t batch, int64_t defaultLen, bool accumulated)
{
    if (seqLengths.empty()) {
        return defaultLen;
    }
    if (accumulated) {
        return batch == 0 ? seqLengths[0] : seqLengths[static_cast<size_t>(batch)] - seqLengths[static_cast<size_t>(batch - 1)];
    }
    return seqLengths[static_cast<size_t>(batch)];
}

int64_t LightningSeqBase(const std::vector<int32_t> &seqLengths, int64_t batch, bool accumulated)
{
    if (!accumulated || seqLengths.empty() || batch == 0) {
        return 0;
    }
    return seqLengths[static_cast<size_t>(batch - 1)];
}

int64_t LightningQueryOffset(const Args &args, int64_t batch, int64_t seq, int64_t head, int64_t dim)
{
    if (args.layoutQuery == "TND") {
        const int64_t base = LightningSeqBase(args.actualSeqLengthsQuery, batch, true);
        const int64_t n = args.queryShape[1];
        const int64_t d = args.queryShape[2];
        return ((base + seq) * n + head) * d + dim;
    }
    const int64_t s = args.queryShape[1];
    const int64_t n = args.queryShape[2];
    const int64_t d = args.queryShape[3];
    return ((batch * s + seq) * n + head) * d + dim;
}

int64_t LightningWeightsOffset(const Args &args, int64_t batch, int64_t seq, int64_t head)
{
    if (args.layoutQuery == "TND") {
        const int64_t base = LightningSeqBase(args.actualSeqLengthsQuery, batch, true);
        const int64_t n = args.weightsShape[1];
        return (base + seq) * n + head;
    }
    const int64_t s = args.weightsShape[1];
    const int64_t n = args.weightsShape[2];
    return (batch * s + seq) * n + head;
}

int64_t LightningKeyHeadNum(const Args &args)
{
    if (args.layoutKey == "PA_BSND") {
        return args.keyShape[2];
    }
    if (args.layoutKey == "TND") {
        return args.keyShape[1];
    }
    return args.keyShape[2];
}

int64_t LightningKeySeqDefault(const Args &args)
{
    if (args.layoutKey == "PA_BSND") {
        return args.keyShape[0] * args.keyShape[1];
    }
    if (args.layoutKey == "TND") {
        return args.keyShape[0];
    }
    return args.keyShape[1];
}

double LightningKeyValue(
    const Args &args, const std::vector<double> &key, int64_t batch, int64_t seq, int64_t head, int64_t dim)
{
    if (args.layoutKey == "TND") {
        const int64_t base = LightningSeqBase(args.actualSeqLengthsKey, batch, true);
        const int64_t n = args.keyShape[1];
        const int64_t d = args.keyShape[2];
        return key[static_cast<size_t>(((base + seq) * n + head) * d + dim)];
    }
    if (args.layoutKey == "PA_BSND") {
        const int64_t blockSize = args.keyShape[1];
        const int64_t n = args.keyShape[2];
        const int64_t d = args.keyShape[3];
        const int64_t maxBlocksPerBatch = args.blockTableShape[1];
        const int64_t blockIndex = seq / blockSize;
        const int64_t blockOffset = seq % blockSize;
        const int32_t physicalBlock =
            args.blockTableValues[static_cast<size_t>(batch * maxBlocksPerBatch + blockIndex)];
        return key[static_cast<size_t>(((static_cast<int64_t>(physicalBlock) * blockSize + blockOffset) * n + head) * d + dim)];
    }
    const int64_t s = args.keyShape[1];
    const int64_t n = args.keyShape[2];
    const int64_t d = args.keyShape[3];
    return key[static_cast<size_t>(((batch * s + seq) * n + head) * d + dim)];
}

int64_t LightningOutputOffset(const Args &args, int64_t batch, int64_t seq, int64_t keyHead, int64_t topk)
{
    const int64_t keyHeads = LightningKeyHeadNum(args);
    if (args.layoutQuery == "TND") {
        const int64_t base = LightningSeqBase(args.actualSeqLengthsQuery, batch, true);
        return ((base + seq) * keyHeads + keyHead) * args.sparseCount + topk;
    }
    const int64_t s = args.queryShape[1];
    return ((batch * s + seq) * keyHeads + keyHead) * args.sparseCount + topk;
}

LightningIndexerGolden GoldenLightningIndexer(
    const Args &args, const std::vector<double> &query, const std::vector<double> &key,
    const std::vector<double> &weights)
{
    const auto outShape = LightningOutputShape(args);
    LightningIndexerGolden golden;
    golden.indices.assign(static_cast<size_t>(Numel(outShape)), -1.0);
    golden.values.assign(static_cast<size_t>(Numel(outShape)), -std::numeric_limits<double>::infinity());

    const int64_t batchSize = LightningQueryBatchSize(args);
    const int64_t querySeqDefault = args.layoutQuery == "TND" ? args.queryShape[0] : args.queryShape[1];
    const int64_t queryHeads = args.layoutQuery == "TND" ? args.queryShape[1] : args.queryShape[2];
    const int64_t headDim = args.layoutQuery == "TND" ? args.queryShape[2] : args.queryShape[3];
    const int64_t keyHeads = LightningKeyHeadNum(args);
    const int64_t groupSize = queryHeads / keyHeads;
    if (keyHeads <= 0 || groupSize <= 0 || queryHeads % keyHeads != 0) {
        throw std::runtime_error("LightningIndexer query heads must be divisible by key heads");
    }

    for (int64_t batch = 0; batch < batchSize; ++batch) {
        const int64_t actQuery =
            LightningActualSeqLen(args.actualSeqLengthsQuery, batch, querySeqDefault, args.layoutQuery == "TND");
        const int64_t actKey = LightningActualSeqLen(
            args.actualSeqLengthsKey, batch, LightningKeySeqDefault(args), args.layoutKey == "TND");
        for (int64_t seq = 0; seq < actQuery; ++seq) {
            int64_t validKeyLen = actKey;
            if (args.sparseMode == 3) {
                validKeyLen = std::max<int64_t>(0, std::min<int64_t>(actKey, actKey - actQuery + seq + 1));
            }
            if (validKeyLen <= 0) {
                continue;
            }
            for (int64_t keyHead = 0; keyHead < keyHeads; ++keyHead) {
                std::vector<std::pair<double, int64_t>> candidates;
                candidates.reserve(static_cast<size_t>(validKeyLen));
                for (int64_t keySeq = 0; keySeq < validKeyLen; ++keySeq) {
                    double score = 0.0;
                    for (int64_t group = 0; group < groupSize; ++group) {
                        const int64_t queryHead = keyHead * groupSize + group;
                        double dot = 0.0;
                        for (int64_t dim = 0; dim < headDim; ++dim) {
                            dot += query[static_cast<size_t>(LightningQueryOffset(args, batch, seq, queryHead, dim))] *
                                   LightningKeyValue(args, key, batch, keySeq, keyHead, dim);
                        }
                        dot = std::max(0.0, dot);
                        score += weights[static_cast<size_t>(LightningWeightsOffset(args, batch, seq, queryHead))] * dot;
                    }
                    candidates.emplace_back(score, keySeq);
                }
                std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
                    if (lhs.first != rhs.first) {
                        return lhs.first > rhs.first;
                    }
                    return lhs.second < rhs.second;
                });
                const int64_t topk = std::min<int64_t>(args.sparseCount, candidates.size());
                for (int64_t idx = 0; idx < topk; ++idx) {
                    const int64_t outOffset = LightningOutputOffset(args, batch, seq, keyHead, idx);
                    golden.indices[static_cast<size_t>(outOffset)] = static_cast<double>(candidates[static_cast<size_t>(idx)].second);
                    golden.values[static_cast<size_t>(outOffset)] = candidates[static_cast<size_t>(idx)].first;
                }
            }
        }
    }
    return golden;
}

NativeResult PrefixNativeResult(NativeResult result, const std::string &label)
{
    if (result.retCode != 0) {
        result.message = label + ": " + result.message;
    }
    return result;
}

NativeResult CompareLightningIndexerResult(
    const std::vector<int32_t> &actualIndices, const std::vector<double> &actualValues,
    const LightningIndexerGolden &golden, bool compareValues)
{
    NativeResult indexResult = CompareResult(actualIndices, golden.indices, false);
    if (indexResult.retCode != 0) {
        return PrefixNativeResult(indexResult, "sparseIndicesOut");
    }
    if (!compareValues) {
        indexResult.message = "golden comparison passed";
        return indexResult;
    }
    NativeResult valueResult = CompareResult(actualValues, golden.values, false);
    if (valueResult.retCode != 0) {
        return PrefixNativeResult(valueResult, "sparseValuesOut");
    }
    valueResult.maxAbsError = std::max(indexResult.maxAbsError, valueResult.maxAbsError);
    valueResult.message = "golden comparison passed";
    return valueResult;
}

template <typename TValue>
NativeResult CompareLightningIndexerResult(
    const std::vector<int32_t> &actualIndices, const std::vector<TValue> &actualValues,
    const LightningIndexerGolden &golden, bool compareValues)
{
    NativeResult indexResult = CompareResult(actualIndices, golden.indices, false);
    if (indexResult.retCode != 0) {
        return PrefixNativeResult(indexResult, "sparseIndicesOut");
    }
    if (!compareValues) {
        indexResult.message = "golden comparison passed";
        return indexResult;
    }
    NativeResult valueResult = CompareResult(actualValues, golden.values, false);
    if (valueResult.retCode != 0) {
        return PrefixNativeResult(valueResult, "sparseValuesOut");
    }
    valueResult.maxAbsError = std::max(indexResult.maxAbsError, valueResult.maxAbsError);
    valueResult.message = "golden comparison passed";
    return valueResult;
}

int64_t RgdrStateOffset(int64_t stateIndex, int64_t head, int64_t valueDim, int64_t keyDim, int64_t nv, int64_t dv, int64_t dk)
{
    return (((stateIndex * nv + head) * dv + valueDim) * dk + keyDim);
}

RecurrentGatedDeltaRuleGolden GoldenRecurrentGatedDeltaRule(
    const Args &args, const std::vector<double> &query, const std::vector<double> &key,
    const std::vector<double> &value, const std::vector<double> &beta, const std::vector<double> &state,
    const std::vector<double> &g, const std::vector<double> &gk)
{
    const int64_t totalTokens = args.queryShape[0];
    const int64_t nk = args.queryShape[1];
    const int64_t dk = args.queryShape[2];
    const int64_t nv = args.valueShape[1];
    const int64_t dv = args.valueShape[2];
    const int64_t group = nv / nk;
    if (group <= 0 || nv % nk != 0) {
        throw std::runtime_error("RecurrentGatedDeltaRule value heads must be divisible by key heads");
    }

    RecurrentGatedDeltaRuleGolden golden;
    golden.out.assign(static_cast<size_t>(totalTokens * nv * dv), 0.0);
    golden.state = state;

    int64_t seq1 = 0;
    for (size_t batch = 0; batch < args.actualSeqLengths.size(); ++batch) {
        const int64_t seq0 = seq1;
        seq1 += args.actualSeqLengths[batch];
        if (seq1 > totalTokens) {
            throw std::runtime_error("RecurrentGatedDeltaRule actualSeqLengths exceed T");
        }
        const int64_t acceptedTokens = args.numAcceptedTokens[batch];
        if (acceptedTokens <= 0 || seq0 + acceptedTokens > seq1) {
            throw std::runtime_error("RecurrentGatedDeltaRule numAcceptedTokens is invalid");
        }
        const int64_t stateBase = args.ssmStateIndices[static_cast<size_t>(seq0 + acceptedTokens - 1)];
        for (int64_t head = 0; head < nv; ++head) {
            const int64_t keyHead = head / group;
            for (int64_t valueDim = 0; valueDim < dv; ++valueDim) {
                std::vector<double> stateRow(static_cast<size_t>(dk), 0.0);
                for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                    stateRow[static_cast<size_t>(keyDim)] =
                        state[static_cast<size_t>(RgdrStateOffset(stateBase, head, valueDim, keyDim, nv, dv, dk))];
                }

                for (int64_t seq = seq0; seq < seq1; ++seq) {
                    const double gamma = args.gPresent ? std::exp(g[static_cast<size_t>(seq * nv + head)]) : 1.0;
                    for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                        stateRow[static_cast<size_t>(keyDim)] *= gamma;
                        if (args.gkPresent) {
                            stateRow[static_cast<size_t>(keyDim)] *=
                                std::exp(gk[static_cast<size_t>((seq * nv + head) * dk + keyDim)]);
                        }
                    }

                    double stateKeyDot = 0.0;
                    for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                        stateKeyDot += stateRow[static_cast<size_t>(keyDim)] *
                                       key[static_cast<size_t>((seq * nk + keyHead) * dk + keyDim)];
                    }
                    const double delta =
                        (value[static_cast<size_t>((seq * nv + head) * dv + valueDim)] - stateKeyDot) *
                        beta[static_cast<size_t>(seq * nv + head)];
                    for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                        stateRow[static_cast<size_t>(keyDim)] +=
                            delta * key[static_cast<size_t>((seq * nk + keyHead) * dk + keyDim)];
                    }

                    double outValue = 0.0;
                    for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                        outValue += stateRow[static_cast<size_t>(keyDim)] *
                                    query[static_cast<size_t>((seq * nk + keyHead) * dk + keyDim)] *
                                    static_cast<double>(args.scaleValue);
                    }
                    golden.out[static_cast<size_t>((seq * nv + head) * dv + valueDim)] = outValue;

                    const int64_t outStateIndex = args.ssmStateIndices[static_cast<size_t>(seq)];
                    for (int64_t keyDim = 0; keyDim < dk; ++keyDim) {
                        golden.state[static_cast<size_t>(RgdrStateOffset(outStateIndex, head, valueDim, keyDim, nv, dv, dk))] =
                            stateRow[static_cast<size_t>(keyDim)];
                    }
                }
            }
        }
    }
    return golden;
}

NativeResult CompareRecurrentGatedDeltaRuleResult(
    const std::vector<HalfStorage<HalfKind::Bf16>> &outHost,
    const std::vector<HalfStorage<HalfKind::Bf16>> &stateHost,
    const RecurrentGatedDeltaRuleGolden &golden)
{
    NativeResult outResult = CompareResult(outHost, golden.out, false);
    if (outResult.retCode != 0) {
        return PrefixNativeResult(outResult, "out");
    }
    NativeResult stateResult = CompareResult(stateHost, golden.state, false);
    if (stateResult.retCode != 0) {
        return PrefixNativeResult(stateResult, "stateRef");
    }
    outResult.maxAbsError = std::max(outResult.maxAbsError, stateResult.maxAbsError);
    outResult.message = "golden comparison passed";
    return outResult;
}

NativeResult RunLightningIndexerWithStream(const Args &args, aclrtStream stream, NativeStageTimings *timings)
{
    NativeResult result;
    TensorHolder queryTensor;
    TensorHolder keyTensor;
    TensorHolder weightsTensor;
    TensorHolder actualSeqLengthsQueryTensor;
    TensorHolder actualSeqLengthsKeyTensor;
    TensorHolder blockTableTensor;
    TensorHolder sparseIndicesTensor;
    TensorHolder sparseValuesTensor;
    void *workspace = nullptr;
    std::vector<double> queryValues;
    std::vector<double> keyValues;
    std::vector<double> weightsValues;
    std::vector<double> ignored;
    LightningIndexerGolden golden;

    try {
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::inputGenerateH2dMs));
            if (CreateGeneratedTensorByDtype(
                    args.dtype, args.queryShape, args.queryRangeLow, args.queryRangeHigh, &queryTensor,
                    &queryValues) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer query tensor failed");
            }
            if (CreateGeneratedTensorByDtype(
                    args.dtype, args.keyShape, args.keyRangeLow, args.keyRangeHigh, &keyTensor, &keyValues) !=
                ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer key tensor failed");
            }
            const std::string weightsDtype = args.weightsDtype.empty() ? args.dtype : args.weightsDtype;
            if (CreateGeneratedTensorByDtype(
                    weightsDtype, args.weightsShape, args.weightsRangeLow, args.weightsRangeHigh, &weightsTensor,
                    &weightsValues) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer weights tensor failed");
            }
        }
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::goldenMs));
            golden = GoldenLightningIndexer(args, queryValues, keyValues, weightsValues);
        }
        auto outShape = LightningOutputShape(args);
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::inputGenerateH2dMs));
            if (!args.actualSeqLengthsQuery.empty() &&
                CreateTensor(args.actualSeqLengthsQuery, {static_cast<int64_t>(args.actualSeqLengthsQuery.size())},
                             aclDataType::ACL_INT32, &actualSeqLengthsQueryTensor) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer actualSeqLengthsQuery tensor failed");
            }
            if (!args.actualSeqLengthsKey.empty() &&
                CreateTensor(args.actualSeqLengthsKey, {static_cast<int64_t>(args.actualSeqLengthsKey.size())},
                             aclDataType::ACL_INT32, &actualSeqLengthsKeyTensor) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer actualSeqLengthsKey tensor failed");
            }
            if (!args.blockTableValues.empty() &&
                CreateTensor(args.blockTableValues, args.blockTableShape, aclDataType::ACL_INT32,
                             &blockTableTensor) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer blockTable tensor failed");
            }

            std::vector<int32_t> sparseIndices(static_cast<size_t>(Numel(outShape)), 0);
            if (CreateTensor(sparseIndices, outShape, aclDataType::ACL_INT32, &sparseIndicesTensor) != ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer sparseIndices output failed");
            }
            const std::vector<int64_t> valuesShape = args.returnValues ? outShape : std::vector<int64_t>{0};
            if (CreateGeneratedTensorByDtype(args.dtype, valuesShape, 0.0, 0.0, &sparseValuesTensor, &ignored) !=
                ACL_SUCCESS) {
                throw std::runtime_error("create LightningIndexer sparseValues output failed");
            }
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto getWorkspaceSize =
            ResolveSymbol<LightningIndexerGetWorkspaceSizeFn>("aclnnLightningIndexerGetWorkspaceSize");
        auto launch = ResolveSymbol<LightningIndexerLaunchFn>("aclnnLightningIndexer");
        auto ret = ACL_SUCCESS;
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::workspaceMs));
            std::string layoutQuery = args.layoutQuery;
            std::string layoutKey = args.layoutKey;
            ret = getWorkspaceSize(
                queryTensor.tensor, keyTensor.tensor, weightsTensor.tensor,
                actualSeqLengthsQueryTensor.tensor, actualSeqLengthsKeyTensor.tensor, blockTableTensor.tensor,
                layoutQuery.data(), layoutKey.data(), args.sparseCount, args.sparseMode, args.preTokens,
                args.nextTokens, args.returnValues, sparseIndicesTensor.tensor, sparseValuesTensor.tensor,
                &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnLightningIndexerGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("allocate LightningIndexer workspace failed: " + std::to_string(ret));
                }
            }
        }
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::launchSyncMs));
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnLightningIndexer failed: " + std::to_string(ret));
            }
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("LightningIndexer stream sync failed: " + std::to_string(ret));
            }
        }

        std::vector<int32_t> actualIndices;
        if (args.returnValues) {
            if (args.dtype == "fp16") {
                std::vector<HalfStorage<HalfKind::Fp16>> values;
                {
                    ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::d2hMs));
                    ret = CopyDeviceToHost(sparseIndicesTensor, outShape, &actualIndices);
                    if (ret == ACL_SUCCESS) {
                        ret = CopyDeviceToHost(sparseValuesTensor, outShape, &values);
                    }
                }
                if (ret == ACL_SUCCESS) {
                    ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::compareMs));
                    result = CompareLightningIndexerResult(actualIndices, values, golden, true);
                }
            } else {
                std::vector<HalfStorage<HalfKind::Bf16>> values;
                {
                    ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::d2hMs));
                    ret = CopyDeviceToHost(sparseIndicesTensor, outShape, &actualIndices);
                    if (ret == ACL_SUCCESS) {
                        ret = CopyDeviceToHost(sparseValuesTensor, outShape, &values);
                    }
                }
                if (ret == ACL_SUCCESS) {
                    ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::compareMs));
                    result = CompareLightningIndexerResult(actualIndices, values, golden, true);
                }
            }
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("copy LightningIndexer sparseValues failed: " + std::to_string(ret));
            }
        } else {
            {
                ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::d2hMs));
                ret = CopyDeviceToHost(sparseIndicesTensor, outShape, &actualIndices);
            }
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("copy LightningIndexer sparseIndices failed: " + std::to_string(ret));
            }
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::compareMs));
            result = CompareLightningIndexerResult(actualIndices, ignored, golden, false);
        }
        if (result.retCode == 0) {
            result.message += "; workspace_size=" + std::to_string(workspaceSize);
        }
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CleanupHolders(
        {&queryTensor, &keyTensor, &weightsTensor, &actualSeqLengthsQueryTensor, &actualSeqLengthsKeyTensor,
         &blockTableTensor, &sparseIndicesTensor, &sparseValuesTensor});
    return result;
}

NativeResult RunRecurrentGatedDeltaRuleWithStream(const Args &args, aclrtStream stream, NativeStageTimings *timings)
{
    NativeResult result;
    TensorHolder queryTensor;
    TensorHolder keyTensor;
    TensorHolder valueTensor;
    TensorHolder betaTensor;
    TensorHolder stateTensor;
    TensorHolder actualSeqLengthsTensor;
    TensorHolder ssmStateIndicesTensor;
    TensorHolder gTensor;
    TensorHolder gkTensor;
    TensorHolder numAcceptedTokensTensor;
    TensorHolder outTensor;
    void *workspace = nullptr;
    std::vector<double> queryValues;
    std::vector<double> keyValues;
    std::vector<double> valueValues;
    std::vector<double> betaValues;
    std::vector<double> stateValues;
    std::vector<double> gValues;
    std::vector<double> gkValues;
    std::vector<double> ignored;
    RecurrentGatedDeltaRuleGolden golden;

    try {
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::inputGenerateH2dMs));
            if (CreateGeneratedTensorByDtype(
                    "bf16", args.queryShape, args.queryRangeLow, args.queryRangeHigh, &queryTensor, &queryValues) !=
                    ACL_SUCCESS ||
                CreateGeneratedTensorByDtype(
                    "bf16", args.keyShape, args.keyRangeLow, args.keyRangeHigh, &keyTensor, &keyValues) !=
                    ACL_SUCCESS ||
                CreateGeneratedTensorByDtype(
                    "bf16", args.valueShape, args.valueRangeLow, args.valueRangeHigh, &valueTensor, &valueValues) !=
                    ACL_SUCCESS ||
                CreateGeneratedTensorByDtype(
                    "bf16", args.betaShape, args.betaRangeLow, args.betaRangeHigh, &betaTensor, &betaValues) !=
                    ACL_SUCCESS ||
                CreateGeneratedTensorByDtype(
                    "bf16", args.stateShape, args.stateRangeLow, args.stateRangeHigh, &stateTensor,
                    &stateValues) != ACL_SUCCESS) {
                throw std::runtime_error("create RecurrentGatedDeltaRule bf16 tensor failed");
            }
            if (CreateTensor(args.actualSeqLengths, {static_cast<int64_t>(args.actualSeqLengths.size())},
                             aclDataType::ACL_INT32, &actualSeqLengthsTensor) != ACL_SUCCESS ||
                CreateTensor(args.ssmStateIndices, {static_cast<int64_t>(args.ssmStateIndices.size())},
                             aclDataType::ACL_INT32, &ssmStateIndicesTensor) != ACL_SUCCESS ||
                CreateTensor(args.numAcceptedTokens, {static_cast<int64_t>(args.numAcceptedTokens.size())},
                             aclDataType::ACL_INT32, &numAcceptedTokensTensor) != ACL_SUCCESS) {
                throw std::runtime_error("create RecurrentGatedDeltaRule int32 tensor failed");
            }
            if (args.gPresent &&
                CreateGeneratedTensorTyped<float>(
                    args.gShape, args.gRangeLow, args.gRangeHigh, aclDataType::ACL_FLOAT, false, &gTensor,
                    &gValues) != ACL_SUCCESS) {
                throw std::runtime_error("create RecurrentGatedDeltaRule g tensor failed");
            }
            if (args.gkPresent &&
                CreateGeneratedTensorTyped<float>(
                    args.gkShape, args.gkRangeLow, args.gkRangeHigh, aclDataType::ACL_FLOAT, false, &gkTensor,
                    &gkValues) != ACL_SUCCESS) {
                throw std::runtime_error("create RecurrentGatedDeltaRule gk tensor failed");
            }
            if (CreateGeneratedTensorByDtype("bf16", args.valueShape, 0.0, 0.0, &outTensor, &ignored) !=
                ACL_SUCCESS) {
                throw std::runtime_error("create RecurrentGatedDeltaRule out tensor failed");
            }
        }
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::goldenMs));
            golden = GoldenRecurrentGatedDeltaRule(
                args, queryValues, keyValues, valueValues, betaValues, stateValues, gValues, gkValues);
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto getWorkspaceSize =
            ResolveSymbol<RecurrentGatedDeltaRuleGetWorkspaceSizeFn>("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize");
        auto launch = ResolveSymbol<RecurrentGatedDeltaRuleLaunchFn>("aclnnRecurrentGatedDeltaRule");
        auto ret = ACL_SUCCESS;
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::workspaceMs));
            ret = getWorkspaceSize(
                queryTensor.tensor, keyTensor.tensor, valueTensor.tensor, betaTensor.tensor, stateTensor.tensor,
                actualSeqLengthsTensor.tensor, ssmStateIndicesTensor.tensor, args.gPresent ? gTensor.tensor : nullptr,
                args.gkPresent ? gkTensor.tensor : nullptr, numAcceptedTokensTensor.tensor, args.scaleValue,
                outTensor.tensor, &workspaceSize, &executor);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(
                    "aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed: " + std::to_string(ret));
            }
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error(
                        "allocate RecurrentGatedDeltaRule workspace failed: " + std::to_string(ret));
                }
            }
        }
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::launchSyncMs));
            ret = launch(workspace, workspaceSize, executor, stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclnnRecurrentGatedDeltaRule failed: " + std::to_string(ret));
            }
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("RecurrentGatedDeltaRule stream sync failed: " + std::to_string(ret));
            }
        }

        std::vector<HalfStorage<HalfKind::Bf16>> outHost;
        std::vector<HalfStorage<HalfKind::Bf16>> stateHost;
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::d2hMs));
            ret = CopyDeviceToHost(outTensor, args.valueShape, &outHost);
            if (ret == ACL_SUCCESS) {
                ret = CopyDeviceToHost(stateTensor, args.stateShape, &stateHost);
            }
        }
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("copy RecurrentGatedDeltaRule output failed: " + std::to_string(ret));
        }
        {
            ScopedStageTimer stage(StageTimingTarget(timings, &NativeStageTimings::compareMs));
            result = CompareRecurrentGatedDeltaRuleResult(outHost, stateHost, golden);
        }
        if (result.retCode == 0) {
            result.message += "; workspace_size=" + std::to_string(workspaceSize);
        }
    } catch (const std::exception &exc) {
        result.status = "failed";
        result.message = exc.what();
        result.retCode = 1;
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CleanupHolders(
        {&queryTensor, &keyTensor, &valueTensor, &betaTensor, &stateTensor, &actualSeqLengthsTensor,
         &ssmStateIndicesTensor, &gTensor, &gkTensor, &numAcceptedTokensTensor, &outTensor});
    return result;
}
