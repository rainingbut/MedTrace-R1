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

## 当前 AutoDL 环境

本项目目前使用 AutoDL 容器实例：

| 项目 | 已核验值 |
|---|---|
| 平台 | AutoDL 容器实例 |
| 容器 UUID | `smyulw5z7c-d584d27f` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 显存 | 24,564 MiB（约 24 GB） |
| NVIDIA Driver | `580.105.08` |
| 驱动显示 CUDA | `13.0` |
| CPU | 16 核 |
| 内存 | 90 GB |
| 系统盘 | 30 GB，基线运行时已用约 9.7 GB |
| 数据盘 | `/root/autodl-tmp`，60 GB |
| Python | `3.12.13` |
| PyTorch | `2.11.0+cu130` |
| Transformers | `5.14.1` |
| vLLM | `0.24.0` |

关键路径：

```text
仓库：        /root/autodl-tmp/MedTrace-R1
虚拟环境：    /root/autodl-tmp/MedTrace-R1/.venv
模型缓存：    /root/autodl-tmp/medtrace-cache/huggingface
运行结果：    /root/autodl-tmp/MedTrace-R1/results
```

大文件、模型、生成数据和 checkpoint 应放在数据盘 `/root/autodl-tmp`，不要放到
30 GB 系统盘。进行大规模 CoT 生成前应再次检查数据盘剩余空间。

## 新开机后的基本操作

AutoDL 关机后磁盘文件保留，但 vLLM、tmux 和其他内存进程都会终止。每次开机后，
先重新进入仓库并激活既有环境：

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate

export VLLM_RUNTIME_BACKEND=native
export HF_HOME=/root/autodl-tmp/medtrace-cache/huggingface
export MEDTRACE_CACHE_DIR="${HF_HOME}"
```

AutoDL 当时访问 Hugging Face, github 等环境，需要先设置：

```bash
source /etc/network_turbo
```

只应在确认目标模型及其固定 revision 已存在于缓存后启用离线模式。不要因为网络问题
静默更换模型或 revision。

最小环境确认：

```bash
git rev-parse HEAD
git status --short

python - <<'PY'
import torch, transformers, vllm
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("cuda_available:", torch.cuda.is_available())
print("bf16_supported:", torch.cuda.is_bf16_supported())
PY

nvidia-smi
df -h / /root/autodl-tmp
```

对于耗时的模型服务、数据生成或验证任务，应放入独立的 `tmux` 会话。通用方式：

```bash
tmux new -s <任务名>
# 在 tmux 内激活 .venv、导出上述环境变量，然后运行任务。
# 脱离：Ctrl+B，松开后按 D
# 重连：tmux attach -t <任务名>
```

每个正式生成阶段都应使用独立、明确命名的输出目录和日志文件。不要使用“最新目录”
猜测目标，不要覆盖既有基线结果。每次开机或服务重启后都应重新捕获对应运行环境；
不得把不同进程、模型 revision、提示版本或解码配置生成的数据混为同一个运行。

步骤二尚未冻结教师模型和验证模型。确定模型后，应另行记录其精确 revision、精度、
上下文长度、并发数、生成参数、缓存路径、服务日志与运行清单，再开始正式生成。

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
