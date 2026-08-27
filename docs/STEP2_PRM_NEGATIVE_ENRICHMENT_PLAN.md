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

## 5. 已批准 canary 与后续关口

### 5.1 第一轮 AutoDL 只读审计结果

2026-08-17 的首次运行得到：

- 自然候选共11条，均为结构合法且仅答案不匹配；
- MedQA 7条，MedMCQA 4条；
- screener reject 候选为0；
- 原 validator 为107条严格正轨迹、1条严格过程负轨迹、1条旧契约不合法轨迹、
  25条不可用响应；
- 6条 recovery canary 均为严格正轨迹；
- 原 PRM 标签为717个整数正标签、4个整数负标签、6个布尔 `true` 标签；
- 4个负 prefix 全部来自同一条负轨迹。

旧契约不合法轨迹与6个布尔标签的数量和步数高度一致。它们源自旧 Python 校验器将
`bool` 视作 `int` 的类型漏洞；不能直接进入严格 SFT/PRM，也不能据此判断医学内容
错误。后续只能选择以下一种可审计处理：

1. 使用 validator v2 对该轨迹重新验证；或
2. 在隔离的派生产物中将布尔标签做有记录的确定性规范化，并保持原产物不变。

首次审计器把“训练产物质量失败”和“源文件完整性失败”合并，因此在成功写出报告后
返回非零。修订版已拆分两个门：只有源文件缺失、计数不一致、键重复或审计期间哈希
变化才抛出异常；严格契约/标签问题会阻止训练就绪，但不会令只读审计命令失败。

### 5.2 已批准 canary

用户已于2026-08-26批准24条 canary，并明确不需要二次批准。冻结配置、预算和
运行命令见 `docs/STEP2_PRM_NEGATIVE_CANARY_RUNBOOK.md`。该授权只覆盖24条
canary，不覆盖合并训练数据或扩大生成。

第一阶段聚合报告完成后，已冻结如下24条 canary 构成：

- 优先 8 条现有自然候选；
- 8 条本地 Qwen2.5-7B gold-blind 学生采样；
- 8 条单点受控难负例；
- MedQA、MedMCQA 各12条。

冻结配额为每路8条、每个来源在 MedQA/MedMCQA 各4条。现存11条自然候选中只
选择4+4条，其余3条 MedQA 候选保留不用。

Canary 最低质量门槛：

- transport/JSON/contract 成功率 100%；
- 至少8条不同的严格负轨迹；
- 人工轨迹标签准确率至少90%；
- 人工首错位置精确准确率至少80%；
- 不得只覆盖单一 benchmark、来源或错误位置。

本24条 canary 和下述3条 HTTP 429 recovery 已获批准；其他 recovery、中等规模
试跑和正式 5,000--15,000 题生成仍需要分别确认。

### 5.3 三条 HTTP 429 定点 recovery

24条 canary 的机器门仅因3条 OpenRouter HTTP 429 无内容响应失败；其余21条严格
契约有效，并已得到11条严格过程负例。用户已于2026-08-27批准只恢复这3条事件：

- 使用相同 `deepseek/deepseek-v4-pro` validator v2；
- 每条最多两次请求，首次/候选间/429重试分别等待60/45/90秒；
- 新增 CNY 2 硬上限、CNY 1.8 停止线；
- 独立 append-only 日志，不覆盖原 canary；
- 不生成新候选、不使用本地 GPU、不合并训练数据、不扩大规模。

准确命令见 `docs/STEP2_PRM_NEGATIVE_VALIDATOR_RECOVERY_RUNBOOK.md`。

### 5.4 结构化人工盲审

3条 recovery 全部首次成功，规范结果达到24/24严格契约、12条严格过程负例，机器
质量门通过。用户已批准纯离线人工盲审评分：24条结构化标注、盲审声明、标注/候选/
原 validator/recovery 哈希锁，以及90%轨迹标签和80%首错位置门。评分仅在人工
`problem_status=ok` 的题目上进行；首错分母进一步限定为人工负轨迹。

双门通过时只能写出“人工有效负例候选元数据清单”，不得写训练记录或合并 PRM。
准确命令见 `docs/STEP2_PRM_NEGATIVE_HUMAN_REVIEW_RUNBOOK.md`。
