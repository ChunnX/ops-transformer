# B=1、T=3500、Nk=16、Nv=32：独立 msopprof 审阅

**结论：主要收益来自减少控制与中间数据往返、缩短Vector工作；Cube实际活跃时间几乎未变。当前是Vector、输入搬运和跨阶段依赖共同限制的融合流水，仍有优化空间，但数据不支持HBM带宽饱和或单一逆矩阵计算占满的结论。**

## 采集与版本

含gate，Dk=Dv=128，Q/K/V及state均BF16，连续state，chunk=64，seed=20260908。两组均使用Default、Occupancy、Roofline，各选1个kernel并重放，工具与pytest退出码均0。每个CSV包含96个唯一角色键：32 cube0、32 vector0、32 vector1。下表按同角色32核求均值；不是32次独立测量。输入形状、dtype、内容SHA256及scale逐项完全相同，二进制hash不同。

采用清洁的baseline_metrics与optimized_metrics，不使用diagnostic_default_vendor_loaded。两组当前/额定频率均1650MHz。正式20次预热+100次测量的msprof结果为898.227→148.505µs，6.048×；本次msopprof的880.146→153.108µs仅用于辅助对照，不替换正式统计。

## 绝对流水时间

|角色/管线|CANN原版 µs/核|当前优化 µs/核|原版/优化|
|---|---:|---:|---:|
|AIC Cube|47.282|47.467|0.996×|
|AIC Scalar|522.942|14.428|36.246×|
|AIC MTE1|45.372|37.498|1.210×|
|AIC MTE2|272.576|47.312|5.761×|
|AIC Fixpipe|353.823|71.344|4.959×|
|AIV0 Vector|322.680|115.938|2.783×|
|AIV1 Vector|273.021|115.711|2.360×|
|AIV0 Scalar|173.699|25.198|6.893×|
|AIV1 Scalar|139.972|25.129|5.570×|
|AIV0 MTE2|198.265|108.587|1.826×|
|AIV1 MTE2|128.967|107.535|1.199×|
|AIV0 MTE3|99.750|27.125|3.677×|
|AIV1 MTE3|71.382|27.096|2.634×|

Cube活跃时间47.282→47.467µs几乎不变，但占比5.388%→31.445%；总Cube指令550→495条/核，仅减少10%。六倍整体收益不能归结为矩阵乘本身快六倍。AIC Scalar减少97.24%，Fixpipe减少79.84%；Vector绝对耗时也显著减少，支持简化控制、片上融合及直接布局转换共同产生收益。这里只能说明机制相符，尚未通过逐项消融精确分摊收益。

原版/优化版AIC ICache miss均值2.600%→0，AIV0/1为4.115%/4.456%→0；控制路径与指令工作集的改善是额外证据，不能独立量化其墙钟贡献。

## 指令与数据路径

|计数（每核）|CANN原版|当前优化|
|---|---:|---:|
|AIC MTE1|1105|1104|
|AIC MTE2|2205|226|
|AIV0 MTE2|723|884|
|AIV1 MTE2|502|878|
|AIV0 MTE3|684|339|
|AIV1 MTE3|519|339|

|全核工具数据计数（KB）|CANN原版|当前优化|
|---|---:|---:|
|read_main_memory_datas(KB)|856327.375|183260.250|
|write_main_memory_datas(KB)|494730.750|32096.000|
|GM_to_L1_datas(KB)|510687.000|56030.250|
|L0C_to_L1_datas(KB)|0.000|63360.000|
|L0C_to_GM_datas(KB)|238748.000|28000.000|
|GM_to_UB_datas(KB)|345640.375|127230.000|

全核读请求计数下降78.60%、写下降93.51%；L0C→GM从238748→28000KB，L0C→L1从0→63360KB，直接反映中间结果改留片上。AIC输入MTE2指令减少约9.76倍，但AIV MTE2指令反而增加：不能笼统写成所有DMA调用都减少。Vector侧部分更多、较小搬运被流水覆盖。上述计数是工具主存请求口径，不能直接写成离芯片HBM流量。

当前AIV GM→UB带宽使用率7.62%/7.56%，AIC GM→L1为6.70%，不足以支持HBM带宽饱和。优化版L2读命中反而降低（AIC83.89%→56.72%，AIV约79%→73%/74%），因此高L2命中不是主要加速原因。

## 当前限制与优化优先级

|指标|原版 AIV0/AIV1|优化 AIV0/AIV1|
|---|---:|---:|
|Vector active|36.738% / 31.070%|76.961% / 76.808%|
|MTE2 active|22.573% / 14.676%|72.080% / 71.380%|
|Vector STU conflict|2.695% / 2.054%|23.824% / 23.681%|
|Vector LDU conflict|9.938% / 9.297%|4.069% / 4.069%|
|Vector wait|28.253% / 20.820%|19.404% / 19.794%|

1. **优先核查Vector写回与输入转换调度。** STU冲突从约2%升至约24%，Vector与MTE2同时繁忙。检查compact NZ写回、Cast/Store、UB区块安排及其与输入DMA的重叠；不能仅凭STU字段就断言是某个bank、某个VF或24%可回收墙钟时间。
2. **继续缩短Vector关键路径，包括逆矩阵、gate转换及state更新。** 当前Vector约116µs/核，比Cube约47µs更接近kernel时长；需要VF级打点或局部消融区分这几部分。SIMD VF内部指令不会在InstrTimeline中展开。
3. **修细已有流水，勿直接复制整套缓冲。** 时间线证明已有大比例V/MTE2重叠，下一步应减少剩余依赖与输入延迟；仅将MTE2占比乘kernel时长当作双缓冲收益不成立。
4. **本例不优先解决head数与尾块。** 32head完整覆盖32组、两个AIV工作接近；各角色核生命周期极差约0.11%，没有显著负载不平衡证据。55块只有最后一块44token，无效20token仅占3520物理token的0.568%，尾块不是数量级空间。

两版Roofline均提示latency bound:pipeline caused。PMU的AIC wait与MTE3 wait在优化后仍偏高，但wait比率不是总墙钟时间分解；不能据其估算成倍潜在收益。

## PipeTimeline独立证据

优化版时间线文件：[trace.json](/home/s00940320/improved_cgdr/attention/chunk_gated_delta_rule/tests/pytest/output/rdv_perf_20260908/network_case_3500/optimized_timeline/prof/OPPROF_20260908073502_LACBZXXAHUTEEOKG/trace.json)，共13194条事件，仅采样core0～5的18个角色轨道。

12个采样AIV的MTE2区间中，85.1%～88.0%已与VECTOR区间重叠。core0.veccore0的MTE2区间99.973µs，其中87.638µs与Vector重叠；Vector区间115.210µs。该结果验证当前实现已有充分的输入/计算并行，无法再将全部输入耗时隐藏一次。

各采样AIC启动阶段有3.73～4.31µs的Cube间隙：例如core0在5.498～9.232µs没有CUBE区间，同时AIV执行输入搬运与Vector工作。稳态仍反复出现更短间隙；其具体来源需要关联符号或消融，不能从pipe名直接分摊给逆矩阵或锁。

本次时间线CUBE为330个连续区间/核、合计约74µs，而Default PMU为495条指令/核、47.47µs；区间定义、合并及采集不同，不能混算。时间线末事件约140µs而OpBasicInfo为153.321µs，亦不能把差值全部解释为可去除等待。trace.json中的ts/dur按Chrome Trace微秒解释，displayTimeUnit=ns只影响显示。

## 口径与异常字段

按同角色分别聚合，NA保留null；ratio保留0～1，带(%)的列保留百分数。总Cube指令可用，但fp/int条数子分类与fp/int cycle ratio不一致，不用于推断实际整数运算。AIC MTE3只有约0.001µs，active_bw因此产生不合理数量级，不解读该字段。

官方说明明确active ratio按各核总周期归一化，wait ratio按指令受阻口径描述；管线可重叠。见[字段说明](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_performance_data.md)。kernel重放会清L2；InstrTimeline不展示SIMD VF内部指令。见[采集指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)。

全部原始均值/最小值/最大值、NA数量、二进制hash、输入hash、时间线区间汇总保存在[msopprof_review.json](msopprof_review.json)。该审阅仅读取采集结果，未运行NPU或修改算子。
