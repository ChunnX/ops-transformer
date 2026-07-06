# A+B 模块设计讨论清单

> 目标: 在开发前将框架骨架(A)和核心数据流(B)的所有设计决策讨论清楚。
> 每个模块列出: 核心问题、待决策项、既有框架参考、当前倾向方案。

---

## A. 框架骨架

### A1. 流程编排器 ✅ 已定稿

**核心问题**: 如何用一个可传入的列表托管 gen/cpu/npu/gpu/compare/npuprof/npuprofop/gpuprof 的任意组合(设计目标15), 不在列表中的步骤不做

**机制**: 阶段注册表 + 管道执行 + CaseContext数据容器
- 阶段注册表: 阶段名→阶段函数, 8个阶段注册, prof阶段为虚函数(raise NotImplementedError)
- 管道执行: 按stages列表顺序逐个执行, 编排器不硬编码顺序
- CaseContext: 持有inputs/outputs/result, 阶段间通过它传递
- 数据依赖自动解决: 每阶段 `ctx.xxx or load_pt(...)`, 内存优先, 无则从pt加载, pt不存在则报错

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 编排器接口签名 | `run_case(stages: list[str], case_id: str, config: Config) -> dict`; 阶段注册表+管道执行+CaseContext; prof阶段注册但抛NotImplementedError | 全量化FA execute_test + 非量化FA run_case |
| 2 | 阶段间数据传递方式 | 内存优先, 默认存pt(设计目标7); 连续阶段内存传递, 跳过阶段从pt加载; 可配置关闭存pt | 非量化FA orchestrator.py:106-123 |
| 3 | 阶段跳过时数据来源 | 每阶段 `ctx.xxx or load_pt(...)`, 有pt则加载, 无则报错 | 全量化FA golden_cache |
| 4 | 阶段定义 | 8个阶段注册于STAGE_REGISTRY, 按stages列表顺序执行; 5个精度阶段有实现, 3个prof阶段虚函数; "带/不带精度校验"=stages中有没有compare | 设计目标(15) |
| 5 | 单case编排 vs 批量编排 | **双入口共享编排器**: pytest驱动(@parametrize批量+pytest.fail不中断) + CLI驱动(单case/调试, for循环+try/except+reporter); 两入口共享run_case和STAGE_REGISTRY | 全量化FA parametrize + 非量化FA脚本循环 |

### A2. 入口脚本

**核心问题**: CLI参数设计、pytest集成方式、conftest钩子边界。充分吸收四框架入口优点。

**四框架优点融合映射**:

| 来源 | 优点 | 融合方式 |
|------|------|----------|
| 全量化FA | golden-mode阶段流水线 | → `--stages` 参数(A1已定) |
| 全量化FA | golden独立CLI | → `{op}_golden.py` 独立`__main__`, `--stages`语义与主入口一致 |
| 全量化FA | SKIP_CASES声明式跳过 | → paramset中定义SKIP_CASES, case_loader自动过滤 |
| 全量化FA | paramset分层(debug/func/perf) | → 用例集文件分层 + `--case-set`选择 |
| 全量化FA | 三层.pt缓存 | → A1#2已定: input.pt/cpu.pt/npu.pt/gpu.pt |
| 全量化FA | msprof conftest编排 | → conftest预留虚函数接口 |
| 全量化FA | --parse-prof事后解析 | → prof模块预留虚函数接口 |
| 非量化FA | --case_id三模式匹配 | → `--case-id`精确/前缀/后缀 |
| 非量化FA | reporter增量CSV+report_interval | → B5直接移植 |
| 非量化FA | 调试三件套 | → `--verbose-diff`/`--visualize`/`--fail-analysis` |
| 非量化FA | 参数互斥校验 | → argparse互斥组 + 启动校验 |
| 非量化FA | 每case资源清理 | → 批量循环中gc.collect()+clear_cache() |
| 非量化FA | 标准退出码 | → sys.exit(0 if fail_cnt==0 else 1) |
| MQSMLA | conftest失败信息定制 | → conftest hookwrapper + monkeypatch terminal |
| MQSMLA | 环境变量传参 | → 环境变量作为YAML配置补充(CI/CD友好) |
| MQSMLA | KEEP_PT中间产物控制 | → `--keep-pt`选项 |
| QSFA | paramset↔Excel互转 | → `--export-excel`/`--import-excel`子命令 |
| QSFA | Excel单元格智能解析 | → case_loader的Excel loader |
| QSFA | 灵活pt输入(文件/目录) | → `--case-id`支持指定pt文件/目录 |

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 入口形式 | **双入口并存, 共享编排器**: pytest为主(parametrize+marker+conftest fixture, CI/CD批量) + CLI为辅(argparse, 单case/golden独立/调试); 两入口参数语义一致, 共享run_case和STAGE_REGISTRY | 全量化FA pytest + 非量化FA脚本 + 全量化FA golden CLI |
| 2 | CLI参数设计 | pytest: conftest addoption(--stages/--case-id/--case-set/--config/--graph-path/--op/--verbose-diff/--visualize/--fail-analysis/--keep-pt/--report-interval); CLI: argparse同语义参数; 环境变量为YAML补充; 参数互斥校验 | 全量化FA conftest addoption + 非量化FA argparse |
| 3 | conftest职责 | **核心层**: addoption定义参数 + session fixture注入(stages→set, config→Config对象) + 失败信息定制(移植MQSMLA hookwrapper+monkeypatch) + msprof编排虚函数(预留) + marker定义(ci/debug/func_rdv/perf_rdv) | 全量化FA conftest fixture + MQSMLA失败信息 + 全量化FA marker |
| 4 | 用例组合与分发 | pytest: @parametrize("params",CASES,ids=CASE_IDS)+SKIP_CASES+marker三层分层(debug/func_rdv/perf_rdv); CLI: --case-id三模式(精确/前缀/后缀)+--case-set指定用例集文件+--import-excel; 环境变量补充 | 全量化FA parametrize+SKIP_CASES + 非量化FA三模式 + QSFA Excel |

### A3. 配置管理 ✅ 已定稿

**核心问题**: 配置结构、全局vs用例级分层、加载时机

**基线**: 全量化FA四件套(paramset_common + --golden-mode + SKIP_CASES + conftest.py)为骨架
**补充**: MQSMLA fill_none_params推导引擎 + 非量化FA InputSpec声明式张量生成

**三层配置结构**:
1. **框架级 config.yaml**: 芯片型号/device_id/graph_path/dtype精度阈值/数据范围/种子/save_pt/report_interval — 全局默认, 启动时加载
2. **算子级 {op}_paramset.py**: PARAM_NAMES(固定顺序) + PARAM_DEFAULTS(默认值) + SKIP_CASES(声明式跳过) — 算子适配定义, import时加载
3. **用例级 字典**: 在paramset文件中定义, 每条用例是字典, 值为list做笛卡尔积展开 — 覆盖框架级默认值

**配置加载与覆盖链**:
```
框架级config.yaml(全局默认)
    ↓ 被覆盖
算子级PARAM_DEFAULTS(算子默认)
    ↓ 被覆盖
用例级字典(具体用例)
    ↓ 传入
normalize_params推导(MQSMLA fill_none_params: 从少量必需参数推导全部参数)
    ↓ 产出
完整params → gen_inputs(非量化FA InputSpec: 声明式tensor生成)
```

**双入口配置注入**:
- pytest入口: conftest session fixture注入(`stages` fixture从--stages解析为set, `config` fixture从--config加载YAML+CLI覆盖)
- CLI入口: argparse解析后直接构造Config对象传入run_case

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 配置文件格式 | 三层: 框架级config.yaml(全局) + 算子级{op}_paramset.py(PARAM_NAMES+DEFAULTS+SKIP_CASES) + 用例级字典 | 全量化FA paramset_common四件套 |
| 2 | 配置内容范围 | 框架级: 芯片/device_id/graph_path/dtype阈值/数据范围/种子/save_pt/report_interval; 算子级: PARAM_NAMES/PARAM_DEFAULTS/SKIP_CASES; 用例级: 具体参数值 | 四框架硬编码项全部纳入配置 |
| 3 | 全局vs用例级覆盖 | 用例级覆盖算子级DEFAULTS, 算子级覆盖框架级config.yaml; 三层merge; case_loader加载时自动合并 | 全量化FA TEST_PARAMS_DEFAULTS + 用例字典覆盖 |
| 4 | dtype精度阈值来源 | 框架级config.yaml中定义(从ops-precision-standard提取), 可按算子级paramset覆盖 | ops-precision-standard阈值表 → YAML |
| 5 | 配置加载时机 | 框架级启动时加载; 算子级import时加载; 用例级case_loader阶段展开笛卡尔积并merge; normalize_params在gen前自动调用; pytest入口通过session fixture注入, CLI入口通过argparse构造 | 全量化FA conftest session fixture |
| 6 | 参数推导引擎 | 移植MQSMLA fill_none_params: 从少量必需参数(如cu_seqlens)推导全部参数(如seqused/block_table); 在case_loader加载后、gen_inputs前调用 | MQSMLA fill_none_params |
| 7 | 声明式张量生成 | 移植非量化FA InputSpec: 建造者模式声明tensor(layout映射+when条件+range_key); 作为gen_inputs的工具, 可选使用 | 非量化FA InputSpec |

---

## B. 核心数据流

### B1. case_loader ✅ 已定稿

**核心问题**: baseparam定义形式、CSV/Excel统一接口、随机测试、ID匹配

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 用例定义形式 | 字典+笛卡尔积为主; CSV/Excel为导入源, 通过--import-excel转为paramset字典, 不作运行时格式 | 全量化FA字典+PARAM_NAMES; QSFA Excel互转 |
| 2 | PARAM_NAMES+DEFAULTS机制 | 算子级定义(A3已定); case_loader框架统一展开: 笛卡尔积+DEFAULTS填充+SKIP_CASES过滤 | 全量化FA paramset_common |
| 3 | 随机测试 | 两层: 轴参数随机(用例字典支持`random(min,max)`语法, case_loader按seed展开) + tensor值随机(gen_inputs内seed控制, InputSpec range_key) | 设计目标(1) + 非量化FA range_key+seed |
| 4 | ID匹配规则 | 精确+前缀+后缀三模式, 直接移植非量化FA case_loader.py:103-120 | 非量化FA resolve_case_ids |
| 5 | CSV/Excel表头 | 列名=PARAM_NAMES, 与字典key一致 | QSFA Excel列名固定 |
| 6 | normalize_params调用时机 | case_loader加载后、gen_inputs前自动调用(A3已定) | MQSMLA fill_none_params |

### B2. gen_data → inputs ✅ 已定稿

**核心问题**: inputs字典key命名规范、InputSpec能力保留、随机种子管理、物理排布传递

**基线**: 非量化FA InputSpec(core/data.py:120-228)为数据生成核心工具, 完整移植其能力

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | inputs字典key命名规范 | 与NPU算子参数名一致, 减少适配层转换; 标量超参(如layout_q/softmax_scale)也放入inputs | 适配方案: inputs含tensor+标量 |
| 2 | InputSpec能力保留 | **完整移植非量化FA InputSpec作为框架标准工具**: 声明式tensor(layout映射+when条件+range_key+layout_override) + 5种生成方法(randn/uniform/fixed/zeros/custom) + 特殊token解析(total_s1/cu_seqlens/num_blocks/_V等) + _resolve轴推导; gen_inputs可选用InputSpec或自由实现 | 非量化FA core/data.py:120-228 |
| 3 | 随机种子管理 | seed作为param, 默认值在DEFAULTS(如seed=42), 可在用例中覆盖; gen_inputs内用torch.Generator+manual_seed, 多tensor共享同一generator保证可复现 | 非量化FA data.py:169-173 seed=42+Generator |
| 4 | inputs字典结构 | inputs含tensor+标量超参, 统一一个字典(适配方案已定); layout作为标量放入inputs(如"layout_q":"BNSD"), CPU/NPU各自按需解读 | 适配方案: 完全一致 |
| 5 | 物理排布tensor的layout传递 | gen_inputs按params中的layout参数产出物理排布tensor; layout_override支持k/v用不同layout(如q=BNSD, kv=PA_BBND); layout标量放入inputs | 非量化FA data.py:178-183 layout_override |
| 6 | gen_inputs与normalize_params关系 | case_loader加载后先normalize_params(推导cu_seqlens→seqused等), 再gen_inputs(轴→tensor); gen_inputs接收已推导的完整params | 非量化FA case_loader.py:26-100 normalize + data.py generate |

### B3. cpu/npu/gpu compute ✅ 已定稿

**核心问题**: 后端抽象、NPU路径封装、GPU可选、outputs结构、异常处理

**基线**: 非量化FA Backend ABC(base.py:8-28)全面领先, 融合全量化FA的graph_path五档语义

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 后端抽象机制 | 移植非量化FA Backend ABC: is_available/compute(inputs,params)->Dict/device property/clear_cache; 算子实现OperatorAdapter继承, 通过装饰器@register_op注册到全局registry; --op选择 | 非量化FA base.py:8-28 |
| 2 | graph_path统一封装 | 框架提供GraphHelper工具类(移植非量化FA _get_graph_net的CompilerConfig封装), 支持graph_path 0/3/5/6/7五档(全量化FA语义); npu_compute接收graph_path参数, 内部调用GraphHelper; 算子无需自己写torchair配置 | 非量化FA npu.py:151-164 + 全量化FA golden.py:1089-1139 |
| 3 | graph_path传入方式 | graph_path作为npu_compute参数, 编排器从config传入(A1已定) | 全量化FA GRAPH_PATH全局变量(反面教材, 不利于并发) |
| 4 | GPU可选 | gpu_compute签名统一(inputs→outputs dict), 实现完全由算子负责; 框架不强制实现, 未实现则is_available返回False; 非量化FA三路径(paged/varlen/fixed)作为参考实现, 不纳入框架核心 | 非量化FA gpu.py:44-54 三路径(参考) |
| 5 | outputs key约定 | out(必选) + 算子自定义可选key(如lse), 注册时声明; 移植非量化FA Dict{"out","lse"}模式 | 非量化FA cpu.py:39, npu.py:210 |
| 6 | 异常处理 | compute抛异常不捕获, 编排器统一try/except(A1#5已定); 不吞异常返回None(QSFA反面教材) | 非量化FA 直接抛 + QSFA吞异常(反面) |
| 7 | device转换 | 编排器统一做: `dev_inputs = {k: v.to(device) if isinstance(v,Tensor) else v for k,v in inputs.items()}`; compute接收已在目标设备的inputs, 内部不做.npu()散布(MQSMLA反面教材) | 非量化FA orchestrator.py:117-118 |
| 8 | clear_cache | 移植非量化FA: Backend基类默认空, NPU子类torch.npu.empty_cache(), GPU子类torch.cuda.empty_cache(); 编排器每case后调用 | 非量化FA base.py:27-28, npu.py:147-149 |

### B4. compare ✅ 已定稿

**核心问题**: 两方/三方判定逻辑、大case并行比对、精度阈值加载

**基线**: 非量化FA compare.py全面领先, 融合MQSMLA的FP8 uint8重解释

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 两方比对判定 | 移植非量化FA check_result: torch.isclose + fail_ratio; 返回dict{passed,max_abs,mean_abs,fail_cnt,total,fail_ratio}; atol/rtol/fail_ratio从config.yaml加载(非硬编码) | 非量化FA compare.py:369-425 |
| 2 | 三方比对判定 | 移植非量化FA Result类(~30指标)+data_compare_benchmark_new(指标级容差倍率判定); 默认两两两方, 可选Result类详细分析; 三方阈值从config加载 | 非量化FA compare.py:45-359 |
| 3 | 大case并行比对 | 可配置: 默认不分块(向量化isclose), 配置chunk_size+num_workers时启用分块多进程并行 | 设计目标(9) |
| 4 | 精度阈值加载 | 框架级config.yaml默认(从ops-precision-standard提取, fp32/fp16/bf16三档) + 算子级get_tolerance覆盖; atol/rtol/fail_ratio/max_error均可配 | 非量化FA get_tolerance(改造为配置驱动) + ops-precision-standard |
| 5 | 比对粒度 | 按outputs key逐个比对(out/lse分别), 失败时触发B6分析 | 非量化FA orchestrator.py:177,204 |
| 6 | compare结果存储 | 结果存csv(增量flush), 失败详情可选存pt | 非量化FA Reporter |
| 7 | FP8/特殊dtype | 移植MQSMLA uint8重解释(float8_e4m3/e5m2→uint8比对) + 非量化FA to_target(dtype映射) + NaN/Inf独立计数+3处理开关(ConfigFmk) | MQSMLA result_compare_method.py:166-195 + 非量化FA compare.py:28-30,310-319 |
| 8 | 底层库 | torch向量化(非numpy), 大tensor性能更好且GPU友好 | 非量化FA torch vs MQSMLA numpy(反面) |

### B5. 精度比对后处理 ✅ 已定稿

**核心问题**: CSV表头规范、增量flush、终端表格、失败结果构造

**基线**: 非量化FA Reporter类全面领先, 融合MQSMLA conftest失败信息定制

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | CSV表头规范 | 通用表头: case,dtype,key,pass,max_abs,mean_abs,fail_ratio,error; 可选扩展列(全params, 配置控制) | 非量化FA reporter.py:24-25 + MQSMLA全params概念 |
| 2 | 增量flush机制 | 移植非量化FA: csv.DictWriter+append+每case flush, O(n); 不用Excel批量写(MQSMLA/QSFA O(n²)反面教材) | 非量化FA reporter.py:21,40-51 |
| 3 | 终端表格格式 | 移植非量化FA Unicode box drawing, 无第三方依赖; 支持中间报告print_interim(--report-interval可配) + 最终汇总print_final | 非量化FA reporter.py:53-95 |
| 4 | 结果收集接口 | 移植非量化FA Reporter.record(name, result, error, dtype); 失败结果构造全NaN占位dict(区分"未跑"与"跑失败0.0"); make_failure_result工厂函数 | 非量化FA reporter.py:30 + test_flash_attn.py:245-250 |
| 5 | 汇总报告 | 终端汇总表(pass/fail/total计数) + 汇总CSV; 无HTML | 非量化FA reporter.py:56-60 |
| 6 | conftest失败信息定制 | 移植MQSMLA: hookwrapper捕获pytest.fail消息 + monkeypatch终端行显示失败原因; 在FAILED后追加显示精度百分比等关键信息 | MQSMLA conftest.py:8-35 |
| 7 | 统一断言 | 移植全量化FA check_results: 按outputs key逐一检查passed字段, 失败时pytest.fail含key+pass率 | 全量化FA test_runner.py:110-123 |

### B6. 精度比对分析 ✅ 已定稿

**核心问题**: 连续性分析、多维度分析、分析报告、可视化

**基线**: 非量化FA analyze_fail_distribution全面领先, 融合全量化FA Max-RE点位

| # | 待决策项 | 定稿方案 | 既有框架参考 |
|---|---------|---------|-------------|
| 1 | 连续性分析 | 移植非量化FA全部5维度: Batch统计(cnt/ratio/max_abs/mean_abs) + Head Top-10 + Seq Top-10 + Batch vs seqused_kv关联 + D维连段(连段数/最长段/平均段长/4模式判定: 全D失败/连段集中/随机分散/混合) | 非量化FA compare.py:428-598 |
| 2 | 分析触发时机 | 可配置: --fail-analysis控制, 默认失败时触发; try/except包裹防分析崩溃 | 非量化FA orchestrator.py:226-239 |
| 3 | 分析报告输出 | txt文件存于data/用例集名/用例名/fail_analysis.txt + 终端打印 | 非量化FA compare.py:591-596 |
| 4 | 失败点位打印 | 融合: verbose_diff可切(默认前10, --verbose-diff全量) + Max-RE点位前3(全量化FA) + 前10后10预览 | 非量化FA compare.py:402-415 + 全量化FA result_compare_method.py:95-104 |
| 5 | 可视化 | 移植非量化FA: 热力图PNG(双段色图, BNSD按panel拆分) + 通过率棒图PNG; --visualize控制; 支持内存tensor入口 | 非量化FA precision_visual.py:139-150,172-287 |
| 6 | seq分段分析 | 移植非量化FA: 按chunk_size分段统计失败率, 可视化序列位置分布 | 非量化FA compare.py:562-571 |

---

## 讨论优先级

按依赖关系排序:

1. **A1 流程编排器** → 定义框架骨架, 所有模块围绕它设计
2. **A3 配置管理** → 定义配置结构, 所有模块从中读取参数
3. **A2 入口脚本** → 定义CLI和conftest边界, 依赖A1/A3
4. **B1 case_loader** → 定义数据入口, 依赖A3(配置)
5. **B2 gen_data** → 定义inputs契约, 依赖B1
6. **B3 compute** → 定义调用契约, 依赖B2
7. **B4 compare** → 定义比对逻辑, 依赖B3
8. **B5 后处理** → 定义结果输出, 依赖B4
9. **B6 分析** → 定义失败分析, 依赖B4
