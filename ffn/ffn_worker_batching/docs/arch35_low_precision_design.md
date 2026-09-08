# FfnWorkerBatching arch35 低精度扩展需求与设计

## 范围与接口

仅为 Ascend950 / DAV_3510（arch35）增加以下 token_dtype，支持 need_schedule=0（batching）及 1（接收并 batching），kernel 和 aclnn 两种调用均覆盖。

| token_dtype | y 数据类型 | dynamic_scale 数据类型 |
| --- | --- | --- |
| 0 | float16 | float32（原有约定） |
| 1 | bfloat16 | float32（原有约定） |
| 2 | int8 | float32 |
| 3 | float8_e5m2 | float8_e8m0 |
| 4 | float8_e4m3fn（用户所指 float8_e4m3） | float8_e8m0 |
| 5 | float4_e2m1 | float8_e8m0 |

输入 schedule_context 仍为 int8 原始上下文，max_out_shape=[A,BS,K,H]，Y=A*BS*K。y 的逻辑形状仍为 [Y,H]，专家排序、mask 压缩、group_list 和四类索引输出保持原语义。搬运必须保持原始位模式，不进行量化、反量化或浮点转换。

存储协议：三种新增类型均为每 32 个逻辑元素一个 E8M0 scale，全部 scale 紧跟整行 token 数据。令 S=ceil(H/32)，dynamic_scale 逻辑形状为 [Y,S]。FP8 token 占 H 字节，FP4 token 占 H/2 字节；每行有效长度 HS 分别为 H+S、H/2+S 字节，源行步长使用上下文中的 attn_to_ffn_token_size（允许行尾 padding）。尾部不足 32 元素的分组仍占一个 scale。

FP4 每字节低/高半字节存放连续两个元素，要求 H 为偶数，以保持每行及多核输出起点按字节对齐；奇数 H 在 host 推导和 arch35 tiling 阶段拒绝。输出只保证 actual_token_num 范围内 token 和 scale 有效，其余尾部不参与比较。

## 实施设计

1. 在 Ascend950 的 OpAICoreConfig 增加完整的六组输入输出签名；ascend910b、ascend910_93 保持三组签名。仅扩展 ascend950 二进制配置。
2. 公共 InferDataType 增加 3/4/5 映射；公共推导函数没有设备路由，以设备注册签名和各架构 tiling 校验约束实际支持范围。
3. arch35 tiling 接受新枚举，保留 tiling key 100/101、工作空间布局和 prepare/sort/group_list 策略。旧架构的枚举校验保持不变。
4. gather 继续按排序后的索引取行，统一用实际 token 字节数规划输出地址和 H 分块。scale 随同行索引搬运，避免 FP4 的 sizeof 类型问题。
5. UB 使用运行时容量，保留索引和 ID 缓冲，token 与新 scale 分块复用搬运队列，不增加随 H 线性增长的缓冲。任务仍按有效 token 行均分到核，块长度由剩余 UB 和双缓冲推导。

## 验证计划

在 tests/st/arch35 现有两份 CSV 扩展新类型，覆盖两种调度模式、mask、重复专家、非块对齐行、大 H 分块；保留 0/1/2 回归。golden 构造原始低精度字节及 scale，验证有效 token 范围内的搬运位模式和所有元数据。

使用 /home/chenpengyi/ops-test-kit 运行 CSV 格式校验、kernel 和 aclnn 测试，记录当前源码构建方式、测试命令、结果及未完成项。host 推导覆盖三组 dtype/shape 和非法参数。


## 缓冲和分块细节

已有 gather 固定缓冲预算为 `19456 + 8*Align(A,8)` 字节（包含双缓冲索引/ID、session 表和向量分道）。令平台返回的可用 UB 为 U，新类型每个 token 队列块大小为 `floor((U-fixed)/2/32)*32` 字节，两块队列共用剩余 UB；scale 搬运不再分配额外缓冲。每行先按块搬 token，再将 `ceil(H/32)` 字节 scale 按相同块大小搬至输出行。队列 EnQue/DeQue 同步 GM→UB→GM，并在 FreeTensor 后复用。

正常和接收模式仍以排序后的有效 token 行均分到核；新类型不改变专家排序的并行度、workspace 和 tiling key。FP4 在分块过程中只处理完整字节，不拆分或重新编码半字节。

## 测试因子与验收

每个新增类型 × need_schedule={0,1} × 8 类场景，共 48 条：最小偶数 H=2、H=66 尾组、多核对齐行、部分 mask、全部 mask、单专家、大 token 行分块、大 scale 行分块。大 scale 行取 H=4194306，scale=131073 字节，确保超过单个搬运块。kernel 保留原有 20 条，aclnn 保留原有 19 条。

Golden 直接构造低精度位模式（包括正负零、FP8 非有限值和全部 E8M0 编码），按原始字节比较有效前缀；FP4 解包后按半字节编码比较。未使用的 FP16/BF16 scale 不比较。kernel 测试使用 snake_case 属性和 numpy 适配器，aclnn 使用 API 原始属性名及 torch 适配器。测试上下文使用现有 TEST_MAGIC 相对偏移协议携带辅助数据。

原 host arch35 用例仍以 A2 的 8 字段 tiling 和固定 32/64 核策略作为期望，本次修正为 arch35 独立结构的语义检查，保留 dtype、调度、UB 容量、大 shape 和大专家数覆盖，并补充工作区不重叠检查。


ACLNN 用例的 schedule_context 输入 shape 为完整打包 buffer 长度，而不是仅 1024 字节的头部长度。当前 M=1 测试中，总长度为 `1024 + Y*HS + 4*Y + 8*A` 字节；FP4 使用 numpy 输入分支，其它类型使用 torch 输入分支，二者写入相同协议。kernel 适配器返回完整原始 buffer。

ACLNN 验证需要包含预编译 kernel 的算子包，应使用 `build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching`；仅 `--jit` 包不满足当前 ACLNN 二进制匹配路径。构建产物安装到独立目录进行测试。


实现和验证已完成：host UT 32/32，kernel ST 68/68，ACLNN ST 67/67 通过。参见 [验证报告](arch35_low_precision_validation.md) 和 [逐用例结果](arch35_low_precision_results.csv)。
