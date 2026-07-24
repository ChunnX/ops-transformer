功能：token attention->FFN
整体文件架构：
attention_to_ffn/
├── op_api/           → 🔌 API层：对外暴露的两阶段接口
├── op_host/          → 🖥️ Host层：算子注册 + 形状推断 + Tiling
├── op_kernel/        → 🧮 Kernel层：AiCore核函
├── op_graph/         → 📊 Graph层：任务生成
├── examples/         → 📝 调用示例
└── tests/            → 🧪 单元测试
examples为调用示例，其步骤为:环境绑定 + 通信域获取-> host数据构造 -> 两阶段API调用 ->资源清理
tests为单元测试，包括各种场景，如量化场景、非量化场景、infershape、tiling（正确性tilingkey验证？、异常校验）
API调用流程：计算workspace大小->创建executor->执行

host层：
定义算子类，public输入、属性等
tiling函数：切块，
tiling执行流程：
AttentionToFFNTilingFunc()
│
├─ 1. CheckAndSetAttrs()          ← 校验属性参数
│    ├─ group/worldSize/moeExpertNum/quantMode/syncFlag/ffnStartRankIdPtr等多个 非空+范围
│    ├─ CheckAndSetShapeAttrs()   ← 校验shape列表维度一致性
│    └─ 设置: tilingData.attentionToFFNInfo.moeExpertNum, quantMode, syncFlag, worldSize...
│
├─ 2. CheckInputAndSetTilingData() ← 校验输入Tensor
│    ├─ quantMode与scales是否匹配 (不支持静态量化)
│    ├─ CheckTensorDim() CheckTensorDataType() CheckTensorFormat()        ← 各Tensor维度数（有几维）、数据类型、格式校验
│    └─ CheckTensorShapeAndSetTinglingData() ← shape值校验（每一维是多少）+填充TilingData（填充tiling结构体的具属性）
│
├─ 3. CalWinSize()                ← 计算窗口大小需求
│    ├─ tokenInfoNeedWinSize = attnWorkerNum × microBatchNum × infoTableLastDim × 4B
│    ├─ tokenDataNeedWinSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × (量化?1B:2B)
│    └─ 校验: neededSize ≤ viableWindowSize
│
├─ 4. SetWorkSpace()              ← 计算workspace大小
│    └─ libApiWorkSpace + ffnNumAlignSize × aivNum
│
├─ 5. SetHcommCfg()               ← 配置HCCL通信
│    └─ AllToAll: level0:fullmesh; level1:pairwise
│
├─ 6. CalTilingKey()              ← 计算TilingKey (3bit组合)
│    └─ [isQuant, isSync, isActiveMask] → 决定kernel模板实例化
│
└─ 7. 设置blockDim = aivNum, 记录ubSize等

kernel层

init：初始化全局和本地tensor
process:执行
重点是SendTokenToFFN，其具体流程为：

1. 准备工作
    包含SplitToCore任务均分，多的前n个核+1

2. 刷新 expertRankTable 的 Cache（确保读到最新值）

3. 循环发送每个 token
    SendTokenToFFNByTokenIdx()，发送单个token

4. [isSync] 把发送状态写回 workspace

SendTokenToFFNByTokenIdx(tokenIdx)
│
├─ 1. 计算当前token的全局偏移 → 拆出 tokenId 和 topkId
│
├─ 2. [isQuant 路径]:
│    ├─ GM → UB: 从x中搬出token数据
│    ├─ SetExpertAndRank(): 查专家路由表
│    ├─ QuantProcess(): 量化为int8
│    ├─ CheckFlagAndSetTableGM(): 等待FFN就绪
│    └─ UB → 远程GM: 写入量化数据和路由信息
│
├─ 2. [!isQuant 路径]:
│    ├─ GM → UB: 从x中搬出token数据
│    ├─ SetExpertAndRank(): 查专家路由表
│    ├─ CheckFlagAndSetTableGM(): 等待FFN就绪
│    └─ UB → 远程GM: 写入原始数据和路由信息
│
└─ 3. 写入 tokenInfoTable 的路由信息 (localExpId)


完整数据流
SendTokenToFFN()
│
├─ SplitToCore(40, 20) → 核0负责offset[0,1], 核1负责[2,3], ...
│
├─ for tokenIdx=0:
│    SendTokenToFFNByTokenIdx(0)
│    │
│    │  [核0, tokenIdx=0]
│    ├─ offset = 0 + 0*20 = 0
│    ├─ tokenId = 0/5 = 0, topkId = 0%5 = 0
│    │
│    ├─ GM → UB: 搬 x[token0] 的 H=7168 个 BF16 元素
│    │
│    ├─ SetExpertAndRank(0, 0, 0)
│    │    ├─ expertIds[0*4+0] = 0 → 路由专家0
│    │    └─ FindExpertRank(0)
│    │         └─ toRank=2, localExp=4
│    │
│    ├─ CheckFlagAndSetTableGM(2)
│    │    └─ 轮询FFN卡2的flag → flag=0 → 就绪
│    │
│    ├─ UB → 远程FFN卡2的窗口:
│    │    tokenData区: 写入 x[token0] 的 7168×2B
│    │    tokenInfoTable区: 写入 localExpId=4
│    │
│    └─ 完成!
│
├─ for tokenIdx=1:
│    SendTokenToFFNByTokenIdx(1)
│    │
│    │  [核0, tokenIdx=1]
│    ├─ offset = 0 + 1*20 = 20
│    ├─ tokenId = 20/5 = 4, topkId = 20%5 = 0
│    ├─ ... (同上流程，发送token4的路由专家0)
│    └─ 完成!
│
└─ 所有token发送完毕



AttentionToFFN 通俗讲解
一个类比：快递分拣中心
想象一家快递公司（MoE模型），它的架构是这样的：

公司有两拨人：
  🏭 分拣中心（Attention Worker）—— 负责接收包裹、决定包裹去哪
  🏠 配送站（FFN Worker）       —— 负责处理包裹（实际计算）

问题：包裹怎么从分拣中心送到正确的配送站？
  → AttentionToFFN 就是这个"派件"过程
完整流程
第1步：包裹到了，每个包裹贴了标签
分拣中心（Attention Worker）已经算完了每个 token，现在每个 token 就像一个包裹，上面贴了一张标签：

Token（包裹）
├─ 内容: 7168个数字（token的特征向量）
└─ 标签: expertIds = [0, 1, 2, 3]
         → 这个包裹应该由 专家0、专家1、专家2、专家3 来处理
         → 外加1个共享专家（每个包裹都要过一遍）
         → 共5份副本要发出去
第2步：查派送地图
分拣员手里有一张派送地图（expertRankTable），上面写着：

专家0 → 部署在 [配送站2, 配送站3, 配送站1]
专家1 → 部署在 [配送站2, 配送站0, 配送站3]
专家2 → 部署在 [配送站1, 配送站3]
专家3 → 部署在 [配送站2, 配送站3, 配送站0, 配送站1, 配送站3, 配送站7]
共享  → 部署在 [配送站1, 配送站4]
分拣员（每个AIV核）根据地图查到：这个包裹的专家0，应该送到配送站2。

第3步：确认配送站有空
发送之前，分拣员要先打电话问一下配送站：

分拣员: "配送站2，你上次那批货处理完了吗？"
配送站: (flag=1) "还没呢，等一下..."
分拣员: (等着...)
...
配送站: (flag=0) "好了，可以发了！"
分拣员: → 开始发货
这就是 CheckFlagAndSetTableGM() 里的 flag 轮询。

第4步：发货
分拣员把包裹（token数据）直接放到配送站的仓库里（远程窗口内存写入）：

分拣员 → 通过网络 → 把数据写到配送站2的仓库
同时 → 在仓库门口的登记表上写: "这个包裹是给专家0的"
       (tokenInfoTable 里的 localExpId)
第5步：通知配送站"你的货到了"
所有包裹都发完之后，分拣员给每个配送站打一个确认电话：

分拣员 → 配送站2: "你的货到了，是第3层模型的，可以开始处理了"
        (SetFlagToFFN: 写入 flag=1 和 layerId)

配送站2: 收到通知 → 开始处理包裹（FFN计算）
为什么要这么做
传统方式（一个FFN处理所有token）:
  Attention → 1个FFN → 慢！大模型参数太多，一个站处理不过来

MoE方式（多个专家分工）:
  Attention → 按标签分发 → 5个配送站同时处理 → 快！并行计算
                ↑
           这就是AttentionToFFN干的事
一张图总结
分拣中心（Attention卡×11）               配送站（FFN卡×5）
┌───────────────────────┐              ┌──────────────────┐
│                       │              │  配送站0  配送站1 │
│ Token0: 标签[0,1,2,3] │──专家0──→   │  配送站2  配送站3 │
│ Token1: 标签[0,1,2,3] │──专家1──→   │  配送站4         │
│ Token2: 标签[0,1,2,3] │──专家2──→   │                  │
│ ...                   │──专家3──→   │  ← 各自处理收到的 │
│ Token7: 标签[0,1,2,3] │──共享──→    │     包裹（FFN计算）│
│                       │              │                  │
│ 查地图 → 确认 → 发货  │              │  收到通知 → 干活  │
└───────────────────────┘              └──────────────────┘

AttentionToFFN = 查地图 + 确认 + 发货 + 通知到货
                 (全是搬运，没有计算)
一句话总结：AttentionToFFN 就是一个智能派件系统——根据路由标签查地图，把每个 token 送到正确的 FFN 专家卡上，仅此而已。

总结：
算子的基本情况，文件结构，写算子要写什么

知识：
isSync，异步


问题
SendTokenToFFN()为什么要刷新expertranktable的cache line？
我还是看不懂attention2fnn的整个流程



