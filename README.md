# MEDTRACE-R1

MEDTRACE-R1 is a personal research project for reproducible medical reasoning
experiments with `Qwen/Qwen2.5-7B-Instruct`. The planned pipeline covers
verified chain-of-thought data construction, supervised fine-tuning, process
reward modelling, GRPO, and selected DAPO mechanisms.

The project is currently at **Stage 0: reproducible baseline evaluation**. No
SFT, PRM, GRPO, or DAPO result is claimed yet.

> This repository is for research and engineering evaluation only. Model
> outputs must not be used as medical advice or as a substitute for qualified
> clinical judgement.

## Current status

| Component | Status |
|---|---|
| Git version control | Complete |
| MedQA/MedMCQA normalisation | Complete |
| Deterministic answer extraction and metrics | Complete |
| Local tests and evaluation dry-run | Complete |
| Qwen2.5-7B-Instruct baseline | Pending GPU run |
| Verified-CoT SFT | Not started |
| PRM | Not started |
| GRPO/DAPO | Not started |

The detailed roadmap is in
[`docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md`](docs/MEDTRACE_R1_IMPLEMENTATION_PLAN.md).

## Reproducible evaluation

The baseline evaluator uses one fixed answer policy:

- The requested output ends with `Final Answer: X`.
- Explicit alternatives such as `the answer is X` can be scored, but are not
  counted as format-valid.
- Conflicting final answers and invalid option labels are scored as incorrect.
- Option-text similarity is never used to guess a missing answer.
- Accuracy, parse rate, and strict format rate are reported separately.

Default decoding is deterministic (`temperature: 0`, `top_p: 1.0`, seed 42).
Every run stores its resolved configuration, input hash, Git revision,
per-question output, latency, parse status, and aggregate metrics.

### 1. Install evaluation dependencies

Use a dedicated environment rather than the legacy HuatuoGPT-o1 training
environment:

```bash
python -m pip install -r requirements/eval.txt
```

### 2. Prepare benchmark files

```bash
python data_pipeline/prepare_benchmarks.py
```

This deterministically creates:

```text
data/benchmark/medqa_test.jsonl
data/benchmark/medmcqa_validation.jsonl
data/benchmark/medical_mcq_eval.jsonl
data/benchmark/manifest.json
```

The generated JSONL files are ignored by Git because they can be reproduced
from the tracked source snapshot. The manifest records input/output SHA256
hashes, sample counts, split names, and current provenance status.

### 3. Run local tests

```bash
python -m unittest discover -s tests -v
```

### 4. Run without a model

```bash
python evaluation/run_eval.py \
  --config configs/eval/qwen2_5_7b_instruct.yaml \
  --input-file tests/fixtures/mini_eval.jsonl \
  --dry-run
```

### 5. Run against a model server

Before a real baseline run, replace `model_revision` in
`configs/eval/qwen2_5_7b_instruct.yaml` with the exact model commit SHA. The
evaluator rejects an unpinned revision for non-dry runs.

Start an OpenAI-compatible vLLM or SGLang server, then run a small pilot:

```bash
python evaluation/run_eval.py \
  --config configs/eval/qwen2_5_7b_instruct.yaml \
  --limit 200
```

After inspecting the pilot outputs, remove `--limit` for the full 5,456-item
evaluation. To resume an interrupted run, pass its directory using
`--run-dir`.

## Repository layout

```text
configs/eval/              Baseline evaluation configuration
data/benchmark/            Generated benchmark files and tracked manifest
data_pipeline/             Deterministic data preparation
docs/                      Roadmap and project documentation
evaluation/                Answer extraction, metrics, and evaluation runner
requirements/              Stage-specific dependency sets
results/baseline/          Local evaluation outputs (ignored by Git)
tests/                     Unit tests and offline fixtures
```

The root-level `SFT_stage1.py`, `RL_stage2.py`, `ppo_utils/`, and data-search
scripts are inherited research code and are retained as references. They are
not yet the MEDTRACE-R1 training implementation. In particular,
`RL_stage2.py` implements PPO rather than the planned GRPO/DAPO pipeline, and
the root `requirements.txt` describes that legacy environment.

## Data provenance and leakage policy

The current evaluation snapshot was inherited from the HuatuoGPT-o1
repository. Its manifest therefore uses the status `inherited_unverified`.
Original dataset revisions and redistribution terms must be verified before a
public release.

MedQA test and MedMCQA validation are evaluation-only in MEDTRACE-R1. They must
not be used for CoT generation, SFT, PRM training, reward-weight selection, or
prompt tuning. Development and reward-development splits will be created from
training data in a later stage.

The inherited `data/demo_data.json` contains examples where generated
reasoning conflicts with the labelled answer. It must not be treated as
verified SFT data.

## Upstream acknowledgement

This repository started from the public
[FreedomIntelligence/HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)
code snapshot and its medical reasoning methodology. MEDTRACE-R1 adds a new
reproducibility, data-verification, SFT, PRM, and GRPO/DAPO engineering track.

The imported snapshot did not include a root license file. Confirm upstream
licensing and dataset terms before redistribution or commercial use.
