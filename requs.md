# 需求 1
1、/home/j60100428/work_doc/build/cov_result/html_report/下面有bash build.sh -u --ophost --ops=sparse_flash_attention --soc=ascend910b --cov的运行结果文件
这个命令是用来查看 lcov 覆盖attention/sparse_flash_attention/op_host下面文件的覆盖率，现在我要你关注的是attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling*的覆盖率，具体用例代码在attention/sparse_flash_attention/tests/ut/op_host/arch32/test_sparse_flash_attention_tiling.cpp里面，当前小点需求你也只能修改这个文件，我要你仿照test_sparse_flash_attention_tiling.cpp的用例继续追加用例以实现更高的覆盖率（可以查看attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling*的实现），不允许你动原来已有的用例（包括改格式）
2、./logs/下面有li_opapi_ut.log和sfa_opapi_ut.log，这个是运行bash build.sh -u --opapi --ops=算子 --cov的失败日志（两个算子：lightning_indexer，简称li；sparse_flash_attention，简称sfa），用例覆盖文件分别为attention/lightning_indexer/tests/ut/op_api/test_aclnn_lightning_indexer.cpp和attention/sparse_flash_attention/tests/ut/op_api/test_aclnn_sparse_flash_attention.cpp，要覆盖的目标是attention/lightning_indexer/op_host/op_api/和attention/sparse_flash_attention/op_host/op_api/，结果是现在两个算子运行api侧的覆盖率都报断错误了，请根据要覆盖的目标排查究竟是哪儿出问题了，只允许你动用例覆盖文件，即attention/lightning_indexer/tests/ut/op_api/test_aclnn_lightning_indexer.cpp和attention/sparse_flash_attention/tests/ut/op_api/test_aclnn_sparse_flash_attention.cpp，可以借鉴attention/prompt_flash_attention/tests/ut/op_api
3、完善用例注释(只允许你动注释，不准动代码)：attention/lightning_indexer/tests/ut、attention/prompt_flash_attention/tests/ut、attention/sparse_flash_attention/tests/ut下面的cpp文件，去完善一下诸如TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_0)或TEST_F(SparseFlashAttentionProto, SparseFlashAttention_infershape_0)前面开头的注释，精简一点（几个单词或缩写描述就行），具体这个用例场景是啥可以借鉴用例参数去算子目录下面的op_host里面文件去对照
注意：详细分析结果和注解在本文档说明

# 需求 2
关于需求 1 的第一小点（仿照test_sparse_flash_attention_tiling.cpp的用例继续追加用例以实现更高的覆盖率）的效果我并不满意，我试了之后看多数新增用例是属于已有的场景，覆盖率只提升了0.1%，现在我说场景，你帮我构造用例
可用用SparseFlashAttention_910b_tiling_7为模板去构造
1、sparse_flash_attention_tiling.cpp 573行的函数：
```
ge::graphStatus SFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
```
OP_CHECK_IF中让判断查找失败条件为真，执行LogErrorDtypeSupport，返回GRAPH_FAILED
2、sparse_flash_attention_tiling.cpp 625行的函数：
```
ge::graphStatus SFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
```
让if (std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) == expectDimNumList.end())为真，执行LogErrorDimNumSupport，返回GRAPH_FAILED
3、sparse_flash_attention_tiling.cpp 656行的函数：
```
ge::graphStatus SFATilingCheck::CheckLayoutSupport(const SFALayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
```
OP_CHECK_IF中让判断查找失败条件为真，执行LogErrorLayoutSupport，返回GRAPH_FAILED
4、sparse_flash_attention_tiling.cpp 512行的函数：
ge::graphStatus SFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const SFATilingShapeCompareParam &param, const SFALayout &layout) const
{
    if (layout == SFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == SFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == SFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else if (layout == SFALayout::BNSG) {
        shapeExpected = gert::Shape({param.B, param.N, param.S, param.G});
    } else if (layout == SFALayout::NTG) {
        shapeExpected = gert::Shape({param.N, param.T, param.G});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", SFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
补充layout == SFALayout::BNSG和layout == SFALayout::NTG和else的场景
5、sparse_flash_attention_tiling.cpp 406行的函数：
```
void SFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum); //balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize);  // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
        if (sfaInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}
```
让if (splitKVFlag_)条件为真