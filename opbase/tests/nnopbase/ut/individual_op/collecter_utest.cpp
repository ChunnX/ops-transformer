/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdio>
#include "utils/indv_types.h"
#include "executor/indv_collecter.h"
#include "executor/indv_bininfo.h"
#include <gtest/gtest.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
#include "register/op_binary_resource_manager.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;

class NnopbaseCollecterUnitTest : public testing::Test {
protected:
    void SetUp() {setenv("ASCEND_C", "1", 1);}
    void TearDown() {unsetenv("ASCEND_C");}
};

void CollecterClean(NnopbaseBinCollecter *collecter) {
    for (size_t i = 0; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; ++i) {
        DList *head = &collecter->regInfoTbl.buckets[i].head;
        for (DoubleListNode *node = head->node.next, *tmp = node->next; node != &(head->node);
             node = tmp, tmp = (node)->next) {
            NnopbaseRegInfo *regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
            if (!regInfo->key.opType.empty()) {
                NnopbaseCollecterGcRegInfo(regInfo);
            }
        }
    }
}

TEST_F(NnopbaseCollecterUnitTest, test_collecter_init_ok)
{
    NnopbaseBinCollecter *bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    delete bin_collecter;
}

TEST_F(NnopbaseCollecterUnitTest, test_get_soc_version_ok)
{
    NnopbaseBinCollecter *bin_collecter = new NnopbaseBinCollecter;
    std::string socVersion = std::string(nnopbase::OPS_SUBPATH_ASCEND910B);
    MOCKER_CPP(&op::PlatformInfo::GetSocVersion).stubs().will(returnValue(op::SocVersion::ASCEND910B));
    nnopbase::IndvSoc::GetInstance().Reset();
    ASSERT_EQ(socVersion, nnopbase::IndvSoc::GetInstance().GetCurSocVersion());
    int32_t ret = NnopbaseSetCollecterSocVersion(bin_collecter);
    ASSERT_EQ(ret, OK);
    delete bin_collecter;
    nnopbase::IndvSoc::GetInstance().Reset();
}

NnopbaseBinCollecter *bin_collecter = NULL;
TEST_F(NnopbaseCollecterUnitTest, test_collecter_work_ok)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_find_binInfo_ok) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    DList *head;
    // opType:"bninference_d_kernel"->key
    uint64_t key = 320;
    head = &bin_collecter->regInfoTbl.buckets[key].head;

    NnopbaseRegInfo *regInfo;
    const char* strKey =  "bninference_d_kernel/d=0,p=0/1,30/1,30/1,30/1,30";
    uint64_t hashKey = 558;
    unsigned char verbose[1024];
    unsigned char *binKey = verbose;
    uint32_t size = 0U;
    ret = NnopbaseCollecterConvertDynamicVerbKey(strKey, binKey, &size);
    ASSERT_EQ(ret, OK);
    NnopbaseBinInfo* binInfoRes;
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        binInfoRes = NnopbaseCollecterFindBinInfo(regInfo, hashKey, verbose, size);
        if (binInfoRes) {
            break;
        }
    }
    ASSERT_NE(binInfoRes, nullptr);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_find_binInfo_fail) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    DList *head;
    uint64_t key = 320;
    head = &bin_collecter->regInfoTbl.buckets[key].head;

    NnopbaseRegInfo *regInfo;
    const char* strKey =  "bninference_d_kernel/d=0,p=0/1,100/1,100/1,100";
    uint64_t hashKey = 558;
    unsigned char verbose[1024];
    unsigned char *binKey = verbose;
    uint32_t size = 0U;
    ret = NnopbaseCollecterConvertDynamicVerbKey(strKey, binKey, &size);
    ASSERT_EQ(ret, OK);
    NnopbaseBinInfo* binInfoRes;
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        binInfoRes = NnopbaseCollecterFindBinInfo(regInfo, hashKey, verbose, size);
        if (binInfoRes) {
            break;
        }
    }
    ASSERT_EQ(binInfoRes, nullptr);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_find_binInfo_hashKey_transborder) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    DList *head;
    uint64_t key = 320;
    head = &bin_collecter->regInfoTbl.buckets[key].head;

    NnopbaseRegInfo *regInfo;
    const char* strKey =  "bninference_d_kernel/d=0,p=0/1,100/1,100/1,100";
    uint64_t hashKey = 1024;
    unsigned char verbose[1024];
    unsigned char *binKey = verbose;
    uint32_t size = 0U;
    ret = NnopbaseCollecterConvertDynamicVerbKey(strKey, binKey, &size);
    ASSERT_EQ(ret, OK);
    NnopbaseBinInfo* binInfoRes;
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        binInfoRes = NnopbaseCollecterFindBinInfo(regInfo, hashKey, verbose, size);
        if (binInfoRes) {
            break;
        }
    }
    ASSERT_EQ(binInfoRes, nullptr);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_bin_info_build_fail_len) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseBinInfo* binInfo = (NnopbaseBinInfo*)malloc(sizeof(NnopbaseBinInfo));
    NnopbaseBinInfoInit(binInfo);
    const char *path = "./built-in/op_impl/ai_core/tbe/kernel/config/ascend910/kernel_bninference_d_dync_main.o";
    unsigned char verbose[25] = {0};
    unsigned int len = 0;
    NnopbaseChar *socVersion = "ascend910b";
    int32_t ret = NnopbaseBinInfoSetOpBinInfoKey(binInfo, verbose, len);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    free(binInfo);
    binInfo = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_read_opinfo_config_fail) {
    bin_collecter = new NnopbaseBinCollecter;
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> commonPath;
    commonPath.push_back(std::make_pair("/usr/local/Ascend/latest/opp/built-in/123/4243", gert::OppImplVersionTag::kOpp));
    int32_t ret = NnopbaseCollecterGetDynamicKernelPathAndReadConfig(bin_collecter, commonPath);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    delete bin_collecter;
    bin_collecter = NULL;
}

TEST_F(NnopbaseCollecterUnitTest, test_convert_verb_key_ok) {
    const char* strKey =  "bninference_d_kernel/d=0,p=0/1,30/1,30/1,30";
    unsigned char verbKey[1024];
    unsigned char *binKey = verbKey;
    uint32_t size = 0U;
    int32_t ret = NnopbaseCollecterConvertDynamicVerbKey(strKey, binKey, &size);
    ASSERT_EQ(ret, OK);
}

TEST_F(NnopbaseCollecterUnitTest, test_get_optype_hashkey_ok) {
    const char opType[] = "bninference_d_kernel";
    size_t expectHashkey = 320;
    size_t hash_key = NnopbaseHashBinary((const unsigned char*)opType, strlen(opType)) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
    ASSERT_EQ(hash_key, expectHashkey);
}

TEST_F(NnopbaseCollecterUnitTest, test_get_verbose_hashkey_ok) {
    const char* strKey = "bninference_d_kernel/d=0,p=0/1,30/1,30/1,30/1,30";
    unsigned char verbKey[1024];
    unsigned char *binKey = verbKey;
    uint32_t size = 0U;
    size_t expectHashkey = 558;
    int32_t ret = NnopbaseCollecterConvertDynamicVerbKey(strKey, binKey, &size);
    ASSERT_EQ(ret, OK);
    uint64_t hashKey = NnopbaseHashBinary(binKey, size) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
    ASSERT_EQ(hashKey, expectHashkey);
}

TEST_F(NnopbaseCollecterUnitTest, test_insert_binCollecter_dlist_ok) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    int node_num = 0;
    uint64_t key = 320;
    NnopbaseRegInfo *regInfo;
    DList *head = &(bin_collecter->regInfoTbl.buckets[key].head);
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        node_num += 1;
    }
    ASSERT_EQ(node_num, 1);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_insert_regInfo_dlist_ok) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    int node_num = 0;
    uint64_t key = 320; // key for bninference_d_kernel
    uint64_t hashKey = 558; // key for one of input of bninference_d_kernel
    DList *head = &(bin_collecter->regInfoTbl.buckets[key].head);
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        NnopbaseRegInfo *regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        if (!regInfo->key.opType.empty()) {
            DList *binhead = &regInfo->binTbl.buckets[hashKey].head;
            for (DoubleListNode *binnode = binhead->node.next; binnode != &(binhead->node); binnode = binnode->next) {
                node_num += 1;
            }
        }
    }
    ASSERT_EQ(node_num, 2); // 1980 & 1971
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_get_tiling_func_ok) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    uint64_t key = 558;
    DList *head = &(bin_collecter->regInfoTbl.buckets[key].head);
    NnopbaseRegInfo *regInfo;
    for (DoubleListNode *node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = ((NnopbaseRegInfo *)((char *)(node) - offsetof(NnopbaseRegInfo, dllNode)));
        ASSERT_NE(regInfo->tiling, nullptr);
    }
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_get_tilingFunc_null) {
    TilingFun tiling;
    NnopbaseJsonInfo jsonInfo;
    jsonInfo.opType = "Add";
    ASSERT_EQ(NnopbaseCollecterSetTiling(jsonInfo, &tiling, gert::OppImplVersionTag::kOpp), OK);
    ASSERT_EQ(tiling, nullptr);
}

TEST_F(NnopbaseCollecterUnitTest, test_load_tilingso_fail) {
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> basePath;
    int32_t ret = NnopbaseLoadTilingSo(basePath);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(NnopbaseCollecterUnitTest, test_get_path_and_read_config_fail) {
    ASSERT_EQ(NnopbaseSetStubNoConfigFiles(), 1);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_get_base_path_fail) {
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
}

TEST_F(NnopbaseCollecterUnitTest, test_find_regInfoInTbl_ok) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    NnopbaseChar opType[50] = "bninference_d_kernel";
    uint64_t hashKey = 320;
    NnopbaseRegInfo* regInfo = NnopbaseCollecterFindRegInfoInTbl(bin_collecter, opType, hashKey);
    ASSERT_NE(regInfo, nullptr);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_find_regInfoInTbl_fail) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    bin_collecter = new NnopbaseBinCollecter;
    int32_t ret = NnopbaseCollecterInit(bin_collecter);
    ASSERT_EQ(ret, OK);
    ret = NnopbaseCollecterWork(bin_collecter);
    ASSERT_EQ(ret, OK);
    NnopbaseChar opType[50] = "bninference_d_kernel";
    uint64_t hashKey = 1024;
    NnopbaseRegInfo* regInfo = NnopbaseCollecterFindRegInfoInTbl(bin_collecter, opType, hashKey);
    ASSERT_EQ(regInfo, nullptr);
    if (!ret) {
        CollecterClean(bin_collecter);
    }
    delete bin_collecter;
    bin_collecter = NULL;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_find_regInfoInTbl_nullptr) {
    bin_collecter = nullptr;
    NnopbaseChar opType[50] = "bninference_d_kernel";
    uint64_t hashKey = 320;
    NnopbaseRegInfo* regInfo = NnopbaseCollecterFindRegInfoInTbl(bin_collecter, opType, hashKey);
    ASSERT_EQ(regInfo, nullptr);
    delete bin_collecter;
}

TEST_F(NnopbaseCollecterUnitTest, test_read_json_fail_1) {
    std::string binaryInfoPath = "";
    nlohmann::json binaryInfoConfig;
    ASSERT_EQ(NnopbaseReadJsonConfig(binaryInfoPath, binaryInfoConfig), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseCollecterUnitTest, test_read_op_json_fail) {
    std::string path = "test";
    std::string path_json_config = path + "/test.json";
    system(("mkdir -p " + path).c_str());
    system(("echo 'test json' > " + path_json_config).c_str());

    nlohmann::json binaryInfoConfig;
    ASSERT_EQ(NnopbaseReadJsonConfig(path_json_config, binaryInfoConfig), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseCollecterUnitTest, test_read_config_file_empty) {
    std::string path = "test";
    std::string path_json_config = path + "/config.ini";
    system(("mkdir -p " + path).c_str());
    system(("echo '#test json' > " + path_json_config).c_str());

    std::vector<std::string> subPath = {};
    ASSERT_EQ(NnopbaseReadConfigFile(path_json_config, subPath), true);
}

TEST_F(NnopbaseCollecterUnitTest, test_update_static_json_info_Fail_1) {
    std::string path = "../satic_kernel/test_staic.json";
    nlohmann::json binInfo;
    NnopbaseJsonInfo jsonInfo;
    ASSERT_EQ(NnopbaseUpdateStaticJsonInfo(binInfo, jsonInfo), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseCollecterUnitTest, test_update_static_json_info_Fail_2) {
    std::string path = "../satic_kernel/test_staic.json";
    nlohmann::json binInfo;
    binInfo["binDesc"]["blockDim"] = 16;
    binInfo["binDesc"]["kernelName"] = "TestStatic_1e04ee05ab491cc5ae9c3d5c9ee8950b";
    binInfo["binDesc"]["workspace"] = {100, 101, 102, 20, 30, 40, 50, 60,100, 101, 102, 20,
        30, 40, 50, 60,100, 101, 102, 20, 30, 40, 50, 60,100, 101, 102, 20, 30, 40, 50, 60};
    NnopbaseJsonInfo jsonInfo;
    ASSERT_EQ(NnopbaseUpdateStaticJsonInfo(binInfo, jsonInfo), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseCollecterUnitTest, test_update_static_bin_json_infos_success) {
    std::string AddTik2Json = "{"
                            "  \"coreType\" : \"VectorCore\","
                            "  \"binList\": ["
                            "    {"
                            "      \"simplifiedKey\": ["
                            "        \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
                            "        \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
                            "      ],"
                            "      \"binInfo\": {"
                            "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_01.json\""
                            "      }"
                            "    },"
                            "    {"
                            "      \"simplifiedKey\": ["
                            "        \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
                            "        \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
                            "      ],"
                            "      \"binInfo\": {"
                            "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_02.json\""
                            "      }"
                            "    }"
                            "  ]"
                            "}";
    std::string AddTik201Json = "{"
                                "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_01.json\","
                                "  \"supportInfo\": {"
                                "    \"simplifiedKey\": ["
                                "      \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
                                "      \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
                                "    ]"
                                "  }"
                                "}";
    std::string AddTik201Bin = "01";
    std::string AddTik202Json = "{"
                                "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_02.json\","
                                "  \"supportInfo\": {"
                                "    \"simplifiedKey\": ["
                                "      \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
                                "      \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
                                "    ]"
                                "  }"
                                "}";
    std::string AddTik202Bin = "02";

    std::vector<std::tuple<const uint8_t*, const uint8_t*>> addTik2OpBinary(
        {{(const uint8_t*)AddTik2Json.c_str(), (const uint8_t*)AddTik2Json.c_str() + AddTik2Json.size()},
        {(const uint8_t*)AddTik201Json.c_str(), (const uint8_t*)AddTik201Json.c_str() + AddTik201Json.size()},
        {(const uint8_t*)AddTik201Bin.c_str(), (const uint8_t*)AddTik201Bin.c_str() + AddTik201Bin.size()},
        {(const uint8_t*)AddTik202Json.c_str(), (const uint8_t*)AddTik202Json.c_str() + AddTik202Json.size()},
        {(const uint8_t*)AddTik202Bin.c_str(), (const uint8_t*)AddTik202Bin.c_str() + AddTik202Bin.size()}});

    nnopbase::OpBinaryResourceManager &manager = nnopbase::OpBinaryResourceManager::GetInstance();
    NnopbaseChar opType[50] = "AddTik2";
    EXPECT_EQ(manager.AddBinary(ge::AscendString(opType), addTik2OpBinary), ge::GRAPH_SUCCESS);
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseBinCollecter collecter;
    EXPECT_EQ(NnopbaseUpdateStaticBinJsonInfos(&collecter, opType), OK);
}

TEST_F(NnopbaseCollecterUnitTest, test_kernel_config) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseBinCollecter collecter;
    std::string basePath = "./usr/local/Ascend/latest/opp";
    nlohmann::json StaticDebugBinInfo;
    StaticDebugBinInfo["coreType"] = 1;
    StaticDebugBinInfo["simplifiedKey"] = "StaticDebug/d=0,p=0/1,3,(20,4,60,60,10)/1,4,(24,4,10,10)/_/_/1,3,(20,4,60,60,10)/[1,1,1,1]/[1,1,1,1]/[1,1,1,1]/1/5748434e/0";
    StaticDebugBinInfo["binPath"] = "ascend910b/StaticDebug_high_performance_0.o";

    nlohmann::json binInfo;
    binInfo["StaticDebug"]["staticList"] = {StaticDebugBinInfo};
    ASSERT_EQ(NnopbaseCollecterReadDebugKernelOpInfoConfig(&collecter, binInfo, basePath, gert::OppImplVersionTag::kOpp), OK);

    nlohmann::json staticBinInfo;
    binInfo["TestStatic"]["binaryList"] = {staticBinInfo};
    ASSERT_EQ(NnopbaseCollecterReadDebugKernelOpInfoConfig(&collecter, binInfo, basePath, gert::OppImplVersionTag::kOpp), OK);

    staticBinInfo["coreType"] = 1;
    staticBinInfo["simplifiedKey"] = "xxxx";
    staticBinInfo["binPath"] = "static_kernel_202307261051/TestStatic/TestStatic_high_performance_0.o";
    binInfo["TestStatic"]["staticList"] = {staticBinInfo};
    ASSERT_EQ(NnopbaseCollecterReadStaticKernelOpInfoConfig(&collecter, binInfo, basePath, gert::OppImplVersionTag::kOpp), OK);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_add_reg_info) {
    NnopbaseRegInfo *reg1;
    NnopbaseBinCollecter collecter;
    NnopbaseCollecterInit(&collecter);
    NnopbaseJsonInfo jsonInfo;
    jsonInfo.opType = "mul";
    ASSERT_EQ(NnopbaseCollecterAddRegInfoToTbl(&collecter, jsonInfo, 1, reg1, gert::OppImplVersionTag::kOpp), OK);

    NnopbaseRegInfo *reg2;
    jsonInfo.opType = "mul";
    ASSERT_EQ(NnopbaseCollecterAddRegInfoToTbl(&collecter, jsonInfo, 1, reg2, gert::OppImplVersionTag::kOpp), OK);
    delete reg1;
    delete reg2;
}

TEST_F(NnopbaseCollecterUnitTest, test_opp_lastest)
{
    ASSERT_EQ(NnopbaseSetStubFiles("", true), 1);
    NnopbaseBinCollecter bin_collecter;
    ASSERT_EQ(NnopbaseCollecterInit(&bin_collecter), OK);
    ASSERT_EQ(NnopbaseCollecterWork(&bin_collecter), OK);
    CollecterClean(&bin_collecter);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCollecterUnitTest, test_read_custom_opapi_path) {
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    std::vector<std::string> basePath;
    NnopbaseGetCustomOpApiPath(basePath);
    ASSERT_NE(basePath.size(), 0);
    NnopbaseUnsetEnvAndClearFolder();

    std::string cust_opp_path = "/usr/local/Ascend/latest/opp/errCust";
    setenv("ASCEND_CUSTOM_OPP_PATH", cust_opp_path.c_str(), 1);
}

TEST_F(NnopbaseCollecterUnitTest, test_read_custom_opapi_path_failed) {
    std::string cust_opp_path = "/usr/local/Ascend/latest/opp/errCust";
    setenv("ASCEND_CUSTOM_OPP_PATH", cust_opp_path.c_str(), 1);
    std::vector<std::string> basePath;
    NnopbaseGetCustomOpApiPath(basePath);
    ASSERT_EQ(basePath.size(), 0);
}