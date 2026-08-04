# Clinical-o1 完整实现方案

## 1. 项目目标

基于 Qwen2.5-7B-Instruct 构建医疗长逻辑推理模型，完成以下闭环：

1. 生成并筛选高质量医疗思维链（CoT）数据；
2. 使用验证后的 CoT 数据进行 SFT；
3. 构造过程监督数据并训练 PRM；
4. 使用复合奖励开展 GRPO，并加入 DAPO 的动态采样等机制；
5. 在 MedQA、MedMCQA 上完成统一评测与消融实验。

最终交付一个可复现的训练项目、模型权重、数据处理流水线、评测报告和可用于简历的项目描述。

---

## 2. 总体技术路线

```text
MedQA/MedMCQA 训练集
        ↓
教师模型生成多条分步 CoT
        ↓
规则检查 + 小模型初筛 + 强模型逐步验证
        ↓
高质量 CoT 数据 + 过程监督数据
        ↓                 ↓
Qwen2.5-7B SFT         PRM 训练
        ↓                 ↓
        └── GRPO/DAPO 复合奖励训练
                        ↓
              自动评测与消融分析
```

基础模型统一使用 `Qwen2.5-7B-Instruct`。资源有限时，SFT 和强化学习均采用 LoRA/QLoRA。

---

## 3. 分步骤实现方案

### 步骤一：建立基线评测

**实现内容**

- 下载并统一 MedQA、MedMCQA 数据格式；
- 固定 train/dev/test，训练过程禁止使用测试集；
- 实现答案抽取、Accuracy 计算和逐题结果保存；
- 评测原始 Qwen2.5-7B-Instruct，记录基线结果。

**输出产物**

- `data/benchmark/`：标准化数据；
- `evaluation/run_eval.py`：统一评测入口；
- `results/baseline.json`：基线结果。

**验收标准**

- 能通过一条命令评测 MedQA 和 MedMCQA；
- 相同参数重复评测结果一致；
- 保存准确率、格式正确率和平均生成长度。

### 步骤二：构造分步 CoT 数据

**实现内容**

- 从训练集选取约 5,000～15,000 道题；
- 使用强教师模型为每题生成 4 条候选 CoT；
- 强制输出 `<step>...</step>` 和 `<answer>...</answer>`；
- 规则检查最终答案、输出格式、重复内容和异常长度；
- 小模型初筛明显错误轨迹；
- 强模型逐步判断事实正确性、逻辑连贯性及最终答案一致性；
- 保留正确轨迹，同时保存“首个错误步骤”作为 PRM 负样本。

**建议数据格式**

```json
{
  "question": "...",
  "answer": "B",
  "steps": [
    {"text": "...", "label": 1},
    {"text": "...", "label": 0}
  ],
  "trajectory_label": 0
}
```

**输出产物**

- `data/cot/sft_verified.jsonl`：正确 CoT；
- `data/prm/process_train.jsonl`：步骤级正负样本；
- `reports/data_statistics.md`：数量、通过率和错误类型统计。

**验收标准**

- 人工抽查至少 300 条；
- 统计步骤标签准确率及两类模型验证的一致率；
- 测试集不得参与生成或筛选。

### 步骤三：进行 Verified-CoT SFT

**实现内容**

- 将当前 SFT 脚本适配 Qwen2.5 Chat Template；
- 只对 assistant 输出计算 loss；
- 支持 LoRA/QLoRA、梯度检查点及断点续训；
- 使用验证后的 CoT 训练 1～3 个 epoch；
- 对比原始模型、普通 CoT SFT 和 Verified-CoT SFT。

**输出产物**

- `train/sft_train.py`；
- `configs/sft_qwen7b.yaml`；
- `checkpoints/sft/`；
- `results/sft_ablation.json`。

**验收标准**

- 训练 loss 正常收敛；
- 输出稳定符合 Thinking/Final Answer 格式；
- Verified-CoT SFT 在验证集优于原始模型，并形成对照表。

### 步骤四：训练过程奖励模型 PRM

**实现内容**

- 使用 Qwen2.5-1.5B/3B 作为 PRM；
- 输入题目和截至当前步骤的推理前缀；
- 输出当前步骤正确概率；
- 使用步骤正负标签进行二分类训练；
- 将整条轨迹的步骤分聚合为过程奖励。

建议采用保守聚合：

```text
process_reward = 0.5 × min(step_scores) + 0.5 × mean(step_scores)
```

**输出产物**

- `train/prm_train.py`；
- `reward/prm_scorer.py`；
- `checkpoints/prm/`；
- `results/prm_metrics.json`。

**验收标准**

- 报告 Accuracy、F1、AUROC；
- PRM 能区分正确轨迹与含中间错误的轨迹；
- 人工样本上完成一次错误步骤定位评估。

### 步骤五：实现 GRPO 复合奖励训练

**实现内容**

- 基于 TRL GRPOTrainer，每道题采样 4～8 个回答；
- 组合以下奖励：

```text
总奖励 = 0.45 × 答案正确性
       + 0.25 × PRM过程分
       + 0.20 × 本地Judge连续分
       + 0.10 × 格式分
       - 重复/超长/答案泄漏惩罚
```

- 本地 Judge 评价医学逻辑、完整性和答案一致性，输出 0～1 连续分；
- 记录组内奖励标准差、KL、Entropy、回答长度和重复率；
- 当 KL 明显超过目标值时增大 KL 系数，过低时减小；
- 设置最大长度和重复惩罚，抑制 Reward Hacking。

**输出产物**

- `train/grpo_train.py`；
- `reward/composite_reward.py`；
- `configs/grpo_qwen7b.yaml`；
- `checkpoints/grpo/`。

**验收标准**

- 组内奖励具有可观察的差异；
- KL 和 Entropy 未发生持续异常；
- 准确率提升不是单纯依赖输出变长或格式投机。

### 步骤六：加入 DAPO 关键机制

本项目只实现与简历描述直接相关的 DAPO 核心机制，不进行大规模完整复现。

**实现内容**

- 动态采样：过滤组内全对、全错或奖励方差过低的样本，并补采新题；
- Clip-Higher：放宽高优势 token 的裁剪上界，维持探索；
- Overlong Shaping：对被截断回答施加平滑惩罚；
- 对比普通 GRPO 与加入 DAPO 机制后的结果。

**输出产物**

- `train/dapo_train.py` 或 GRPO Trainer 扩展；
- `configs/dapo_qwen7b.yaml`；
- `checkpoints/dapo/`；
- `results/rl_ablation.json`。

**验收标准**

- 记录动态采样过滤比例；
- 对比 GRPO 与 DAPO 的准确率、KL、Entropy 和奖励曲线；
- 至少完成一次去除 PRM、去除动态采样的消融实验。

### 步骤七：最终自动化评测

**实现内容**

- 使用完全相同的 prompt、解码参数和答案抽取器评测所有模型；
- 对比 Base、SFT、GRPO、DAPO 四个阶段；
- 保存逐题预测，并分析诊断错误、知识错误、推理错误和格式错误；
- 至少使用 3 个随机种子报告平均值和标准差。

**最终结果表**

| 模型 | MedQA | MedMCQA | 格式正确率 | 平均长度 |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 待实测 | 待实测 | 待实测 | 待实测 |
| Verified-CoT SFT | 待实测 | 待实测 | 待实测 | 待实测 |
| SFT + GRPO | 待实测 | 待实测 | 待实测 | 待实测 |
| SFT + DAPO | 待实测 | 待实测 | 待实测 | 待实测 |

只有实测并保存配置、日志和 checkpoint 后，才能把具体提升数字写入简历。

---

## 4. 最小目录改造

```text
Clinical-o1/
├── configs/             # SFT、GRPO、DAPO 配置
├── data/
│   ├── benchmark/       # 标准化 benchmark
│   ├── cot/             # 验证后的 CoT
│   └── prm/             # 步骤监督数据
├── data_pipeline/       # 生成、验证和过滤脚本
├── train/               # SFT、PRM、GRPO、DAPO
├── reward/              # 格式、答案、PRM、Judge 奖励
├── evaluation/          # 统一评测与答案抽取
├── results/             # 指标和消融结果
└── reports/             # 数据统计与最终报告
```

---

## 5. 推荐实施周期

| 周期 | 工作内容 |
|---|---|
| 第1天 | 数据标准化、基线评测、答案抽取器 |
| 第2天 | 多轨迹 CoT 生成与规则筛选 |
| 第3天 | 大小模型逐步验证、PRM 数据构造 |
| 第4天 | Verified-CoT SFT 与对照实验 |
| 第5天 | PRM 训练和过程奖励接入 |
| 第6天 | GRPO 训练、复合奖励和监控 |
| 第7天 | DAPO 动态采样、消融实验 |
| 第8天 | 最终评测、图表、README 和简历整理 |

资源有限时，可将数据量缩小并采用 LoRA，但不得省略基线、数据隔离和消融实验。

---

## 6. 最终交付清单

- 一套可复现的数据生成与过程验证脚本；
- Verified-CoT 数据集及步骤级 PRM 数据集；
- SFT、PRM、GRPO、DAPO 四类训练脚本和配置；
- Base/SFT/GRPO/DAPO 模型或 LoRA 权重；
- MedQA、MedMCQA 自动评测结果；
- KL、Entropy、Reward、长度等训练曲线；
- PRM 与动态采样消融实验；
- 项目 README、技术报告和简历项目描述。

---

## 7. 最终简历描述模板

### Clinical-o1：基于 Qwen2.5-7B 的医疗长逻辑推理与过程监督对齐模型

**项目负责人｜个人项目｜起止时间按实际填写**

**项目简介：** 面向复杂医疗诊断与病例分析场景，基于 Qwen2.5-7B-Instruct 构建从 Verified-CoT 数据、过程奖励模型到 GRPO/DAPO 强化学习的完整对齐闭环，提升模型在医疗长逻辑推理任务中的准确率与输出规范性。

**主要工作：**

- 设计分步 CoT 数据生成链路，引入规则过滤、小模型初筛和强模型逐步验证，构造带步骤正确性及首错位置标签的过程监督数据，并训练医疗 PRM；
- 完成 Verified-CoT SFT，对比普通 CoT 与过程验证数据的训练效果，增强模型的长逻辑诊断和规范作答能力；
- 实现 GRPO/DAPO 强化学习管线，设计“答案正确性 + PRM 过程分 + 本地 LLM Judge 连续分 + 格式约束”的复合奖励，通过动态采样、KL 调节、Entropy 监控及超长惩罚抑制奖励投机；
- 搭建 MedQA、MedMCQA 自动化评测与消融体系，对比 Base、SFT、GRPO、DAPO 各阶段效果，最终将 MedQA 从 **[基线]%** 提升至 **[实测]%**，MedMCQA 从 **[基线]%** 提升至 **[实测]%**。

> 注意：方括号中的指标必须替换为真实实验结果；如果只实现了部分 DAPO 机制，应写“引入 DAPO 的动态采样与 Clip-Higher 机制”，不要写成“完整复现 DAPO”。
