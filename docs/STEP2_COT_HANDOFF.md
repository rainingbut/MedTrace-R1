# MEDTRACE-R1 步骤二交接：分步 CoT 数据

## 当前状态

步骤一“建立基线评测”已经完成并验收。首个官方完整基线使用：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- revision：`a09a35458c702b33eeacc393d103063234e8bc28`
- Git：`8459db5504723a8ac59f152d376582597c8aa288`
- 数据：MedQA test 1,273 条；MedMCQA validation 4,183 条
- 总计：5,456 条，5,456 个唯一 ID，无重复、API 错误或截断
- MedQA accuracy：`0.630008`
- MedMCQA accuracy：`0.560602`
- Overall accuracy：`0.576796`
- Overall parse/format rate：`0.996884`
- 总耗时：`1309.98179` 秒

私有备份：

```text
medtrace-baseline-8459db5-20260809-final.tar.gz
SHA256: 8AF8AC60FB8832E307E19B16F281E271C3EFDA1B54C9EC1C843FE7841449BD96
```

归档和解压后的逐题预测必须保持私有，不要提交到 Git。

## 下一步

进入实施计划的步骤二“构造分步 CoT 数据”。目标产物：

- `data/cot/sft_verified.jsonl`
- `data/prm/process_train.jsonl`
- `reports/data_statistics.md`

必须遵守：

1. 只使用训练集生成和筛选 CoT；测试集不得参与。
2. 每题生成 4 条候选轨迹，格式为 `<step>...</step>` 与 `<answer>...</answer>`。
3. 依次进行规则过滤、小模型初筛和强模型逐步验证。
4. 保留正确轨迹，并记录首个错误步骤作为 PRM 负样本。
5. 正式生成前先完成小规模端到端试跑和成本估算。
6. 数据中记录来源 split、模型/revision、提示版本、解码参数和验证结果，保证可追溯。

## 第一个决策关口

开始写代码或调用教师模型前，先确定：

- 可合法使用的 MedQA/MedMCQA 训练 split 及许可边界；
- 强教师模型、初筛模型和逐步验证模型；
- 本地部署还是 API，以及预算和隐私要求；
- 首轮试跑规模（建议 20–50 题，每题 4 条）；
- 轨迹 schema、提示版本和失败分类。

不要在这些项目未冻结前启动 5,000–15,000 题的正式生成。

## 下一会话提示词

```text
继续 MEDTRACE-R1 的步骤二。先完整阅读
docs/STEP2_COT_HANDOFF.md 和 docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md。
官方完整基线已经验收并私有备份。先检查仓库现状和训练数据隔离，带我完成
CoT 数据 schema、教师/验证模型、提示、成本与 20–50 题试跑方案的决策关口；
未经确认不要启动大规模生成，也不要使用测试集构造训练数据。
```
