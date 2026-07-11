/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "register/op_def_registry.h"
namespace ops {
class CausalTopoGCN : public OpDef {
public:
    explicit CausalTopoGCN(const char *name) : OpDef(name) {
        // 【TODO: 需要你修改】请核对你的 GCN 算子真实的输入和属性名字！
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}); // 假设有邻接矩阵
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        
        // 假设的属性，请根据实际情况增删
        this->Attr("in_channels").Int(); 
        this->Attr("out_channels").Int();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(CausalTopoGCN);
} // namespace ops
