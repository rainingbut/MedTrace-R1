# MEDTRACE-R1 新会话交接：Step 2 CoT 试跑后质量审计

更新时间：2026-08-27（Asia/Shanghai）

## 0. 2026-08-27 最新状态（优先于下文旧状态）

下文保留了 40 题审计前后的历史操作记录。本节是当前工作的最新入口：

- 已完成的24条 canary 实现基线为 `819b3b6`；recovery 以当前远端分支最新提交为准；
- validator v2 recovery 已修复原25条失败响应的空内容/JSON契约问题；
- 6条 recovery canary 全部完成且结构合法，但全部判为正；
- 原 pilot PRM 严格标签约为723正、4负，尚未达到 PRM 训练就绪状态；
- 根因是原候选漏斗在强验证前排除了答案错误和 screener reject 候选；
- 用户已批准 PRM 负样本增广的第一阶段离线工作；
- 第一阶段不包含模型调用、API费用、GPU推理、剩余19条 recovery 或新 CoT 生成。
- 第一轮负例机会审计发现11条自然候选：MedQA 7条、MedMCQA 4条，均为答案不匹配；
- screener reject 自然候选为0，因此11条现存候选不足以单独解决负样本问题；
- 另发现1条旧契约不合法 canonical，对应6条布尔 `true` PRM 标签；
- 修订版将源完整性与训练质量分离，不再因已报告的标签质量问题令审计命令失败；
- 当前本地完整单元测试为100项全部通过。
- 用户已于2026-08-26批准固定24条 PRM 负样本 canary，且不需要二次批准；
- canary 固定为现有自然/本地学生/单点受控错误各8条，每路 MedQA/MedMCQA 各4条；
- OpenRouter 硬上限 CNY 20、停止线 CNY 18，本地 GPU 上限1小时；
- 准确命令见 `docs/STEP2_PRM_NEGATIVE_CANARY_RUNBOOK.md`；
- 该批准不包含将 canary 合并进训练数据或扩大规模。
- 24条 canary 已完成：21条严格契约、11条严格过程负例、3条 validator unavailable；
- 3条 unavailable 全部为 OpenRouter HTTP 429 且没有响应内容，不是数据或契约错误；
- 用户已于2026-08-27批准只恢复这3条事件，每条最多两次，新增预算硬上限 CNY 2、
  停止线 CNY 1.8；
- recovery 不需要本地 GPU，不得覆盖原 canary，准确命令见
  `docs/STEP2_PRM_NEGATIVE_VALIDATOR_RECOVERY_RUNBOOK.md`。
- 3条 recovery 已全部首次成功；规范审计达到24/24严格契约和12条严格过程负例，
  integrity/machine quality 均通过，combined cost 为 CNY 1.91987452；
- 用户已批准纯离线结构化人工盲审评分；准确命令见
  `docs/STEP2_PRM_NEGATIVE_HUMAN_REVIEW_RUNBOOK.md`；
- 人工门通过前不得生成训练记录；通过后也只允许候选元数据清单，不允许合并。

当前设计与命令见：

```text
docs/STEP2_PRM_NEGATIVE_ENRICHMENT_PLAN.md
configs/cot/prm_negative_enrichment_v1.yaml
data_pipeline/audit_prm_negative_opportunities.py
schemas/prm_negative_candidate_v1.schema.json
configs/cot/prm_negative_validator_recovery_v1.yaml
docs/STEP2_PRM_NEGATIVE_VALIDATOR_RECOVERY_RUNBOOK.md
configs/cot/prm_negative_human_review_v2.yaml
docs/STEP2_PRM_NEGATIVE_HUMAN_REVIEW_RUNBOOK.md
configs/cot/prm_negative_human_adjudication_v1.yaml
docs/STEP2_PRM_NEGATIVE_HUMAN_ADJUDICATION_RUNBOOK.md
```

原始 `results/cot/pilot_v1_real/` 七类核心产物必须保持不变。公开聚合审计不得输出
题目 ID、题干或轨迹文本。24条 canary 的构成和预算已经冻结并获批，可按
`docs/STEP2_PRM_NEGATIVE_CANARY_RUNBOOK.md` 直接执行，不需要二次确认。

## 1. 新会话必须先做什么

新会话开始后，请先完整阅读：

1. `docs/STEP2_POST_PILOT_HANDOFF.md`（本文件，当前状态的唯一入口）
2. `docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md`（项目总路线）
3. `docs/STEP2_DECISIONS.md`（Step 2 已冻结决策）
4. `docs/STEP2_PILOT_RUNBOOK.md`（40 题试跑的完整运行方法）

然后检查本机和 AutoDL 的 Git 状态。不要重新生成 40 题，不要启动更大规模
CoT 生成，也不要把私有结果提交到 Git。

给新会话的建议首条消息：

```text
继续 MEDTRACE-R1 Step 2。请先完整阅读
docs/STEP2_POST_PILOT_HANDOFF.md、docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md、
docs/STEP2_DECISIONS.md 和 docs/STEP2_PILOT_RUNBOOK.md。
40 题正式 CoT 试跑已经在 AutoDL 完成；当前下一步是运行并分析只读质量审计，
不是继续生成。每次进入新的执行阶段前先等待我的确认。
```

## 2. 项目目标

MEDTRACE-R1 的目标是基于 `Qwen/Qwen2.5-7B-Instruct` 构建可复现的医疗长逻辑
推理训练闭环：

1. 从 MedQA、MedMCQA 的训练 split 生成并验证分步 CoT；
2. 构造 Verified-CoT SFT 数据；
3. 构造步骤级正负样本并训练 PRM；
4. 进行 Verified-CoT SFT；
5. 使用复合奖励实施 GRPO，并引入必要的 DAPO 机制；
6. 在隔离的 MedQA、MedMCQA 评测集上统一评估 Base/SFT/GRPO/DAPO；
7. 保存可复现代码、配置、运行清单、私有模型产物与聚合报告。

当前只推进到 Step 2。Step 3 及以后尚未授权启动。

## 3. 已完成状态

### 3.1 Step 1 官方完整基线

Step 1 已完成、验收并私有备份：

- 模型：`Qwen/Qwen2.5-7B-Instruct`
- revision：`a09a35458c702b33eeacc393d103063234e8bc28`
- 基线运行 Git：`8459db5504723a8ac59f152d376582597c8aa288`
- MedQA test：1,273 题，accuracy `0.630008`
- MedMCQA validation：4,183 题，accuracy `0.560602`
- 总题数：5,456，overall accuracy `0.576796`
- overall parse/format rate：`0.996884`
- 私有备份：`medtrace-baseline-8459db5-20260809-final.tar.gz`
- 备份 SHA-256：
  `8AF8AC60FB8832E307E19B16F281E271C3EFDA1B54C9EC1C843FE7841449BD96`

基线逐题预测和备份必须保持私有。

### 3.2 Step 2 数据隔离

训练源已固定并验证：

- MedQA train 原始 10,178 条，隔离后 10,172 条；
- MedMCQA train 原始 182,822 条，隔离后 182,774 条；
- 原始总数 193,000；接受 192,946；拒绝 54；
- 拒绝项包含 13 条规范化评测重叠和 41 条高相似重叠；
- 接受集 ID 和 content SHA-256 均唯一；
- 禁止使用评测 split、`evaluation/data/eval_data.json` 和
  `data/demo_data.json` 构造训练数据。

私有训练源与清单位于 AutoDL 的 `data/source/`，由 `.gitignore` 排除。

### 3.3 冻结的 CoT 方案

| 角色 | 模型/版本 | 关键约束 |
|---|---|---|
| 教师 | DashScope `qwen3-max-2026-01-23` | `teacher_v1`，gold-blind，4 次独立请求 |
| 初筛 | 本地 `Qwen/Qwen2.5-7B-Instruct` | revision `a09a...bc28`，vLLM 0.24.0 |
| 强验证 | OpenRouter `deepseek/deepseek-v4-pro` | `validator_v1`，ZDR，禁止数据收集 |

OpenRouter 额外约束：

- `reasoning.effort=high`；
- 同一精确模型 slug 内允许供应商 fallback；
- `require_parameters=true`；
- `sort=price`；
- `max_price` 为 USD 2.10/M input、USD 4.40/M output；
- 使用 OpenRouter 返回的 `usage.cost` 记真实费用；
- 缺失 `usage.cost` 时失败关闭；
- API 总硬上限 CNY 10，程序停止线 CNY 9。

### 3.4 40 题正式试跑（已完成）

试跑输入：

- 20 MedQA train + 20 MedMCQA train；
- 每题 4 个教师候选，共 160 个；
- 固定 seed：`20260811`；
- 题目文件 SHA-256：
  `eb7f4c2ba71d4651929af89558bfa64c295dcdab902e218483f931c040df9f76`；
- 配置 SHA-256：
  `168243919b2912ba76b1e70d4228d011cd18b2a2d3583ed9f026b712677ca36f`；
- 正式生成运行的 Git commit：
  `461a95cb28837f8072b6fc3d927f57fd8fb80f87`。

正式运行结果位于 AutoDL：

```text
/root/autodl-tmp/MedTrace-R1/results/cot/pilot_v1_real/
```

完成结果：

| 指标 | 数量 |
|---|---:|
| questions | 40 |
| teacher events | 160 |
| rule passed | 134 |
| screener events | 134 |
| validator events | 134 |
| canonical trajectories | 109 |
| SFT records | 108 |
| PRM prefix records | 727 |
| 实际总费用 | CNY 7.26784155 |

`metadata.json` 状态为 `complete`，费用低于 CNY 9 停止线。

重要初步信号（尚需审计确认）：

- 134 个 validator event 只形成 109 条 canonical，推测约 25 条验证请求或
  JSON/契约处理未完整成功；
- 109 条 canonical 中 108 条为 SFT 接受，负轨迹可能严重不足；
- 727 条 PRM prefix 的正负比例尚未汇总；
- 尚未判断 screener verdict、validator 错误类型、题目覆盖和候选重复度。

不要把上述推测当成最终结论；必须以审计报告为准。

### 3.5 当前 Git 状态

- 工作分支：`step2_genCOTdata`
- 当前远端/本机期望 HEAD：
  `900ef5101492cf646d6257f054ce90e2b47b80fa`
- 当前提交：`Add aggregate CoT pilot quality audit`
- 上一提交 `461a95c` 是正式 40 题生成快照；
- `900ef51` 只新增只读审计器及测试，不改变已经生成的数据。

审计代码：

```text
data_pipeline/audit_cot_pilot.py
tests/test_audit_cot_pilot.py
```

本机已通过 62 项单元测试，但 AutoDL 还需要拉取 `900ef51` 后再执行审计。

## 4. 当前唯一已授权的下一阶段

运行 40 题结果的只读质量审计并分析报告：

1. 同步 `900ef51`；
2. 运行 62 项测试；
3. 对 `results/cot/pilot_v1_real/` 执行聚合审计；
4. 检查结构一致性、费用一致性和以下分布：
   - teacher 状态、重试和规则失败代码；
   - screener pass/review/reject、错误代码；
   - validator 完成/失败、重试、错误类别、provider；
   - validator 正负轨迹标签、首错位置、problem status；
   - PRM prefix 正负标签和错误代码；
   - 40 题至少获得一条 canonical/SFT 的覆盖率；
   - 同题 4 个候选的精确重复和近重复；
   - teacher、validator 和总费用是否与 metadata 一致；
5. 根据审计结果提出修正方案和下一个决策关口。

审计不调用模型、不需要 API Key、不需要 GPU 推理。

## 5. 尚未授权的工作

以下操作必须在审计完成、给出结论并获得用户新的明确确认后才能执行：

- 修复 validator prompt/schema/retry 并重新调用失败样本；
- 补采负轨迹或修改筛选策略；
- 将 pilot 输出复制为正式 `data/cot/sft_verified.jsonl` 或
  `data/prm/process_train.jsonl`；
- 扩展到 5,000–15,000 题或任何更大规模生成；
- 启动 Step 3 SFT；
- 启动 PRM、GRPO 或 DAPO 训练；
- 发布、再分发或商业使用题目及衍生轨迹。

每次进入新的执行阶段前，都要等待用户确认。

## 6. 项目环境

### 6.1 Windows 本地开发环境

```text
仓库：D:\Projects\LLM\MedTrace-R1
分支：step2_genCOTdata
Python：D:\Miniconda3\python.exe
GPU：RTX 3050 Laptop，4,096 MiB（不能运行批准的 7B screener）
```

本地用于代码编辑、单测、schema 和私有输入复核，不用于正式模型服务。

Windows PowerShell 启动检查：

```powershell
cd D:\Projects\LLM\MedTrace-R1
git branch --show-current
git rev-parse HEAD
git status --short
D:\Miniconda3\python.exe -m unittest discover
```

预期分支为 `step2_genCOTdata`，HEAD 为 `900ef51...`，工作区为空。

### 6.2 AutoDL 正式环境

当前使用的实例标识（2026-08-12 运行时）：

```text
容器 shell：autodl-container-2a5a4cb260-61932923
仓库：/root/autodl-tmp/MedTrace-R1
虚拟环境：/root/autodl-tmp/MedTrace-R1/.venv
模型缓存：/root/autodl-tmp/medtrace-cache/huggingface
GPU：NVIDIA GeForce RTX 4090，24,564 MiB
Python：3.12.13
PyTorch：2.11.0+cu130
Transformers：5.14.1
vLLM：0.24.0
```

AutoDL 关机后磁盘文件保留，但 tmux 和 vLLM 进程会终止。大文件、模型、私有
数据和 checkpoint 应留在 `/root/autodl-tmp` 数据盘。

AutoDL 普通终端通用初始化：

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
source /etc/network_turbo

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

git branch --show-current
git rev-parse HEAD
git status --short
nvidia-smi
df -h / /root/autodl-tmp
```

## 7. 新会话下一步的准确命令

### 7.1 AutoDL 同步只读审计器

先确认工作区没有非忽略修改：

```bash
cd /root/autodl-tmp/MedTrace-R1
git status --short
```

然后同步：

```bash
git fetch origin \
  +refs/heads/step2_genCOTdata:refs/remotes/origin/step2_genCOTdata

git merge --ff-only origin/step2_genCOTdata

git rev-parse HEAD
git status --short
```

预期 HEAD：

```text
900ef5101492cf646d6257f054ce90e2b47b80fa
```

若不能 fast-forward，不要使用 `reset --hard`，先检查分支和提交图。

### 7.2 AutoDL 运行测试

```bash
source .venv/bin/activate
python -m unittest discover
```

预期 62 项测试通过。

### 7.3 AutoDL 执行只读质量审计

不需要启动 vLLM，也不需要注入 DashScope/OpenRouter Key：

```bash
python -m data_pipeline.audit_cot_pilot \
  --run-dir results/cot/pilot_v1_real \
  2>&1 | tee results/cot/pilot_v1_real/quality_audit.log
```

输出文件：

```text
results/cot/pilot_v1_real/quality_audit.json
results/cot/pilot_v1_real/quality_audit.md
```

查看并复制回新会话：

```bash
cat results/cot/pilot_v1_real/quality_audit.md
python -m json.tool results/cot/pilot_v1_real/quality_audit.json
```

报告只包含聚合统计，不包含题目 ID、题干或轨迹文本。

## 8. 各终端/tmux 启动与恢复指令

### 8.1 查看现有 tmux

```bash
tmux ls
```

40 题已完成，质量审计不需要任何 tmux 服务。为节省 GPU 费用，可以停止旧服务。

### 8.2 `cot-screener`：本地 vLLM 初筛服务

仅在未来获准进行模型调用时才需要。启动：

```bash
tmux new -s cot-screener
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"

./scripts/serve_vllm.sh 2>&1 | tee results/cot-screener-vllm.log
```

脱离：`Ctrl+B`，松开后按 `D`。恢复：

```bash
tmux attach -t cot-screener
```

停止服务：进入 tmux 后按 `Ctrl+C`，再输入 `exit`。

每次代码提交变化后，正式调用前必须重新捕获 runtime：

```bash
python scripts/capture_runtime.py \
  --backend native \
  --vllm-version 0.24.0 \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output results/runtime/cot_screener_runtime_manifest.json
```

不得使用 `--allow-dirty` 生成正式 runtime manifest。

### 8.3 `cot-pilot-40`：已完成的正式试跑任务

这项任务已经完成，不应重新运行。查看旧会话：

```bash
tmux attach -t cot-pilot-40
```

查看完成日志：

```bash
tail -n 100 results/cot/pilot_v1_real/runner.log
python -m json.tool results/cot/pilot_v1_real/metadata.json
```

只有在明确修复并获准续跑时，才重新在同一 tmux 内输入 Key：

```bash
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY && echo
export DASHSCOPE_API_KEY
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY && echo
export OPENROUTER_API_KEY
export MEDTRACE_API_KEY=EMPTY
```

环境变量只存在于设置它的 shell；普通终端 `export` 不会自动进入已存在的 tmux。
绝对不要把 Key 写进 YAML、日志、脚本或 Git。

### 8.4 可选：停止旧 tmux/关机

审计不需要 GPU。如果确认 40 题任务已经完成，可停止 screener：

```bash
tmux send-keys -t cot-screener C-c
```

建议进入会话确认进程已退出，不要盲目删除私有结果目录。AutoDL 关机前检查：

```bash
tmux ls
git status --short
df -h /root/autodl-tmp
```

## 9. 私有数据和 Git 安全边界

下列目录已被 `.gitignore` 排除，必须保持私有：

```text
data/source/
data/cot/
data/prm/
results/cot/
results/verification/
results/runtime/
```

提交前必须检查：

```bash
git status --short
git diff --cached --name-only
git diff --cached --check
```

不得提交或公开：

- MedQA/MedMCQA 私有训练题；
- 教师原始响应；
- canonical、SFT、PRM JSONL；
- per-record validator/screener 事件；
- API Key、`.env`、runtime 私有信息；
- 基线逐题预测和私有备份。

当前许可边界是私有、非商业研究。发布、再分发或商业使用需要单独权利审查。

不要删除 `results/cot/pilot_v1_real/`。它是当前唯一正式 40 题试跑结果。

## 10. 审计后的决策关口

新会话收到 `quality_audit.md/json` 后，应先给出证据化分析，再等待用户确认下一阶段。
至少需要回答：

1. 25 条非 canonical validator event 的确切原因是什么？
2. 是 API/路由/JSON 契约问题，还是医学验证拒绝？
3. 108/109 SFT 接受是否说明 validator 过松？
4. PRM 的 `label=0` 数量是否足够支撑 PRM 训练？
5. 40 题的 SFT 覆盖率是否均衡？
6. 四候选是否具有足够多样性？
7. 是否需要仅重试失败验证、修改 prompt/schema，或刻意补采负轨迹？
8. 当前 40 题结果能否作为扩大规模前的可信模板？

可能的后续分支（都需重新确认）：

- A：只修 validator 失败并对失败事件做受控续跑；
- B：收紧 validator 或增加负轨迹采样，重新做小规模对照试跑；
- C：当前方案通过质量关口，设计正式生成规模和成本预算；
- D：停止 Step 2，优先人工抽查和标注规范修订。

即使工程审计通过，也不能直接进入大规模生成；原计划还要求正式数据阶段进行人工抽查，
最终至少抽查 300 条，并报告步骤标签准确率和模型验证一致率。

## 11. 不要做的事情

- 不要重新 clone 并删除当前 AutoDL 仓库；私有 Git-ignored 数据只存在 AutoDL。
- 不要使用 `git reset --hard`、`git clean -fdx` 或递归删除仓库。
- 不要删除事件日志以绕过 resume/config/hash 错误。
- 不要静默替换模型、revision、prompt、供应商或价格上限。
- 不要让评测集参与生成、筛选、SFT 或 PRM。
- 不要因审计代码提交从 `461a95c` 前进到 `900ef51`，就把生成数据的来源提交改写为
  `900ef51`；生成快照仍是 `461a95c`。
- 不要在没有新确认的情况下产生新的 API 费用或启动 Step 3。

## 12. 关键文件索引

| 文件 | 用途 |
|---|---|
| `configs/cot/pilot_v1.yaml` | 冻结的模型、采样、预算和输出配置 |
| `data_pipeline/run_cot_pilot_real.py` | 正式 40 题生成/续跑器 |
| `data_pipeline/audit_cot_pilot.py` | 当前下一步：只读聚合质量审计 |
| `data_pipeline/cot_preflight.py` | GPU、Key、模型和 runtime 预检 |
| `data_pipeline/cot_prompts.py` | teacher/screener/validator 提示版本 |
| `data_pipeline/cot_rules.py` | 教师 XML 与答案规则过滤 |
| `data_pipeline/cot_isolation.py` | 训练/评测隔离逻辑 |
| `schemas/cot_trajectory_v1.schema.json` | canonical trajectory contract |
| `docs/STEP2_DECISIONS.md` | 已冻结决策 |
| `docs/STEP2_PILOT_RUNBOOK.md` | 40 题试跑运行手册 |
| `docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md` | 全项目实施路线 |
