# 精度比对方法未注册

## 现象

```
KeyError: 'cv_fused_double_benchmark'
```

headDim 问题解决后跑 id=61，pyaclnn 和 cpu 阶段正常，精度比对阶段崩溃。

## 根因

`aclnnFusedFloydAttention.yaml` 指定：

```yaml
standard:
  acc:
    cv_fused_double_benchmark:
      max_re_ration: 10
      avg_re_ratio: 2
      root_mean_squared_ratio: 2
```

但当前 ATK 环境的 `ACCURACY_REGISTRY` 只注册了三个比对器：
`equal`、`single_bm`、`default`，**没有** `cv_fused_double_benchmark`。

## 修复

**文件**：`aclnnFusedFloydAttention.yaml:11`（及 json 对应字段）

将比对方法改为已注册的 `single_bm`（支持相对误差阈值，与原配置语义最近）：

```yaml
standard:
  acc:
    single_bm:
      max_re_ration: 10
      avg_re_ratio: 2
      root_mean_squared_ratio: 2
```

改完后同步重新生成 json，或直接改 json 里的对应字段。

> 若项目确实需要 `cv_fused_double_benchmark` 的自定义逻辑，需额外提供比对器插件并通过 `-p` 注册到 `ACCURACY_REGISTRY`。
