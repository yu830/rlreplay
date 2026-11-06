# 中文快速开始

RLReplay 的总目标是构建开放、可扩展、面向生产环境的LLM训练正确性验证与故障重放基础设施。当前版本聚焦固定verl/TRL配置下的outcome-GRPO，提供采集、独立检查、语义重放，以及verl原生重放与回归导出。TRL另有显式的[冻结生成/奖励边界原生回放](../trl-native-replay.md)，使用包含文本和token的独立包，不改变默认数值采集。[总Goal](../../GOAL.md) · [支持范围](../compatibility.md)

先从源码目录运行CPU合成样例。需要Python 3.11或以上版本和uv。

```powershell
uv sync --locked
$demo = Join-Path ([IO.Path]::GetTempPath()) ("rlreplay-demo-" + [guid]::NewGuid())
uv run --locked python examples/frozen_contract.py $demo
$healthyCapsule = (Get-ChildItem -LiteralPath (Join-Path $demo 'healthy') -Filter '*.json').FullName
$faultyCapsule = (Get-ChildItem -LiteralPath (Join-Path $demo 'faulty') -Filter '*.json').FullName
uv run --locked python -m rlreplay inspect $healthyCapsule
uv run --locked python -m rlreplay check $healthyCapsule
uv run --locked python -m rlreplay check $faultyCapsule
uv run --locked python -m rlreplay replay $faultyCapsule
uv run --locked python -m rlreplay diff $healthyCapsule $faultyCapsule
```

样例生成健康和故障两份capsule，并打印文件位置。健康检查返回PASS和退出码0，故障检查及语义重放返回SUSPECTED_MISMATCH和退出码2。`diff`报告发生变化的样本，完成比较后退出码为0。PowerShell用`$LASTEXITCODE`读取最近一条命令的退出码。

Linux/WSL可采用对应命令。

```bash
uv sync --locked
demo_dir=$(mktemp -d)
uv run --locked python examples/frozen_contract.py "$demo_dir"
healthy_capsule=$(find "$demo_dir/healthy" -name '*.json' -type f)
faulty_capsule=$(find "$demo_dir/faulty" -name '*.json' -type f)
uv run --locked python -m rlreplay check "$healthy_capsule"
uv run --locked python -m rlreplay check "$faulty_capsule"
uv run --locked python -m rlreplay replay "$faulty_capsule"
uv run --locked python -m rlreplay diff "$healthy_capsule" "$faulty_capsule"
```

| 退出码 | 含义 |
|---|---|
| 0 | PASS，或inspect/diff/export操作完成 |
| 1 | CONTRACT_VIOLATION，明确契约违反 |
| 2 | SUSPECTED_MISMATCH，独立数值检查发现差异 |
| 3 | INCONCLUSIVE，证据或执行资格不足 |
| 4 | 输入、源码版本或运行环境错误 |

语义重放重新计算独立参考并检查捕获结果。对真实verl driver采集包，可以在具备匹配依赖的训练环境中执行原生重放及回归导出。

```bash
python -m rlreplay replay /path/to/verl-capsule.json --runner verl
python -m rlreplay export-test /path/to/verl-capsule.json ./new-regression
python -m pytest ./new-regression/test_replay.py
```

原生重放核对已记录的框架源码和运行时指纹。验证候选修复时可显式使用`--allow-revision-change`。导出的回归测试启用了候选验证，运行前先阅读其内容。已知错误会持续触发测试失败，直到候选实现满足声明的契约。

接入实际训练时，verl使用`VerlObserver.around()`，TRL使用`GRPOCaptureMixin`。先确认支持版本和配置，再提供生产者身份、独立输出目录和采样预算。训练结束后关闭Recorder，并读取完成、丢弃、错误和检查回执。

[架构](../architecture.md) · [接入指南](../adapters.md) · [运行与数据管理](../production.md) · [遥测](../telemetry.md) · [验证](../validation.md)

开发环境的完整检查应启用可选数值格式依赖。

```bash
uv sync --locked --dev --extra binary
uv run --locked --extra binary pytest
uv run --locked --extra binary ruff check .
```
