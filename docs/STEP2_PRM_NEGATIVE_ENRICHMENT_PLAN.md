# Step 2 PRM 负样本增广方案

更新时间：2026-08-17（Asia/Shanghai）

## 1. 当前问题

40 题正式 CoT 试跑得到 727 条 PRM prefix records，但严格标签约为 723 正、
4 负。validator v2 recovery 已解决原先响应为空或 JSON 不合法的问题；6 条
recovery canary 均成功且结构合法，但全部判正，因此 recovery 不是负样本增广
方案。

根因是原 pilot 的候选漏斗为 SFT 正样本质量而设计：

- `gold_answer_mismatch` 会令 teacher 规则检查失败；
- 只有规则通过的候选进入 screener；
- 只有 screener `pass/review` 进入 validator；
- 强 teacher 产生的剩余候选本身高度偏正。

现有 SFT 正轨迹不因此失效，但当前 PRM 数据未达到训练就绪状态。

## 2. 不可变边界

- 原始 `results/cot/pilot_v1_real/` 七类核心产物保持不变；
- 只使用已隔离的 MedQA、MedMCQA `train` split；
- SFT 继续只接收完整验证通过的正轨迹；
- 疑似负例在独立 validator 验证前没有训练标签；
- 私有候选、题目、轨迹和逐条判断保持 Git-ignored；
- 任何 GPU、教师、validator 或其他模型调用都需要新的明确批准。

机器可读政策位于：

```text
configs/cot/prm_negative_enrichment_v1.yaml
```

私有候选契约位于：

```text
schemas/prm_negative_candidate_v1.schema.json
```

契约故意不包含 `label` 字段，候选只能标记为
`requires_independent_validation`。

## 3. 标签语义

PRM 当前学习的是“截至当前步骤的整个 prefix 是否仍然正确”，不是孤立步骤文本的
真假：

- 首错之前：`prefix_label=1`；
- 首个明确错误步骤及其后续 prefix：`prefix_label=0`；
- 严格负例要求 `problem_status=ok`、`trajectory_label=0`、首错位置有效，且
  首错步骤的 `local_verdict=incorrect`；
- 首错为 `uncertain` 的轨迹进入人工复核；
- `ambiguous`、`bad_gold` 进入人工复核；
- 只有答案不一致、但没有错误推理步骤的轨迹不作为 PRM step negative。

训练就绪度必须按“具有明确首错的独立轨迹数”衡量。不能通过同一条早期错误轨迹
产生的大量后续零标签 prefix 制造表面类别平衡。

## 4. 第一阶段：已批准的离线工作

第一阶段不进行模型调用，内容为：

1. 冻结上述政策与私有候选契约；
2. 对原 pilot 做只读、聚合级负例机会审计；
3. 统计以下自然候选，但不自动赋负标签：
   - 结构合法且唯一失败原因是 `gold_answer_mismatch`；
   - screener verdict 为 `reject`；
4. 汇总现有 validator、recovery、canonical 和 PRM 的严格标签证据；
5. 验证审计前后七类源产物 SHA-256 完全不变；
6. 输出不含题目 ID、题干或轨迹文本的 JSON/Markdown 聚合报告。

AutoDL 执行命令：

```bash
python -m data_pipeline.audit_prm_negative_opportunities \
  --config configs/cot/prm_negative_enrichment_v1.yaml
```

输出：

```text
results/cot/pilot_v1_real/prm_negative_enrichment_v1/prm_negative_opportunity_audit.json
results/cot/pilot_v1_real/prm_negative_enrichment_v1/prm_negative_opportunity_audit.md
```

该命令不读取 API Key、不访问模型端点、不需要 GPU。

## 5. 后续关口（尚未授权）

收到第一阶段聚合报告后，才能冻结确切 canary 构成和预算。当前建议目标为24条：

- 优先 8 条现有自然候选；
- 8 条本地 Qwen2.5-7B gold-blind 学生采样；
- 8 条单点受控难负例；
- MedQA、MedMCQA 各12条。

以上是目标配额而非已授权运行。自然候选不足时必须先根据审计结果调整配额，不能
静默替换来源。

Canary 最低质量门槛：

- transport/JSON/contract 成功率 100%；
- 至少8条不同的严格负轨迹；
- 人工轨迹标签准确率至少90%；
- 人工首错位置精确准确率至少80%；
- 不得只覆盖单一 benchmark、来源或错误位置。

Canary、剩余 recovery、中等规模试跑和正式 5,000--15,000 题生成均需要分别确认。
