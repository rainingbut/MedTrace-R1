# Step 2: formal scale-up planning and zero-call budget precheck

Approval: offline planning approved on 2026-08-29. This phase reads the completed
private pilot/canary/recovery/materialization logs and writes only aggregate quota
and budget reports. It does not authorize a paid canary, model/API calls, vLLM,
GPU inference, data generation, training, Git push, or full-scale execution.

## Frozen planning policy

- Preserve four independent gold-blind teacher candidates per new question.
- Balance every new-question quota equally between MedQA train and MedMCQA train.
- Report four total negative-trajectory targets: 200 (checkpoint), 300 (minimum
  trainable prototype), 500 (formal recommendation), and 1,000 (more robust
  optional tier). The current 12 negative trajectories count toward each total.
- Guarantee new negative quotas only through controlled-single-error (80%) and
  local-student (20%) routes. Existing teacher answer mismatches remain
  opportunistic and must still pass independent validation and human review.
- Convert observed 8-case origin yields to candidate quotas with a conservative
  one-sided Wilson lower bound (`z=1.281551565545`), rather than the point rate.
- Retain every strict Verified-CoT trajectory for SFT. For PRM only, stratify the
  newly available positive-prefix pool to target a 5:1 positive/negative ratio;
  acceptable final planning bounds remain 3:1–8:1.
- Derive provider request counts, tokens, and costs from raw durable ledgers. Use
  1.15x for the budgeted estimate, 1.25x for the program stop line, and 1.40x for
  the absolute hard cap.
- Treat local GPU hours as a conservative cap-scaled upper bound because the old
  event logs do not contain measured GPU wall time. The documented CNY 2.5/hour
  rate is an observation, not a future price guarantee, and must be reconfirmed.

Machine-readable policy is in `configs/cot/step2_scaleup_plan_v1.yaml`.

## Data-product roles and formal scale target

- The estimated 6,687.5 strict Verified-CoT trajectories from the recommended
  2,500 new questions are retained for SFT. They teach the policy model to emit
  correct, structured medical reasoning.
- PRM training consumes prefix-level binary labels. Negative prefixes teach
  first-error detection; sampled positive prefixes calibrate correct-prefix
  scores. Prefixes from one trajectory are correlated, so trajectory count—not
  raw prefix count—is the primary diversity measure.
- The 200-trajectory tier is a checkpoint, not a final PRM claim. The formal
  recommendation is 500 total negative trajectories. With observed step lengths,
  it projects to roughly 1,955 negative prefixes and 9,775 sampled positive
  prefixes at the frozen 5:1 target, or about 11,700 PRM records total.
- Canary cases and held-out human-review trajectories measure pipeline and label
  quality. They are not automatically authorized as training inputs.
- Split SFT trajectories 95%/5% into train/validation and PRM negative
  trajectories 80%/10%/10% into train/validation/test. The split unit is the
  source question, never an individual prefix. For the 500-trajectory target,
  PRM quotas are therefore 400/50/50. The current 12 negatives default to the
  audit holdout portion unless a later training-data audit authorizes otherwise.

## Required private inputs

The following completed artifacts must remain under
`results/cot/pilot_v1_real/` on AutoDL:

- pilot `quality_audit.json` plus teacher, screener, and validator event ledgers;
- strict 107-record SFT view;
- 24-candidate canary, both local generation ledgers, and validator ledger;
- three-event recovery ledger and canonical recovery audit;
- locked adjudication aggregate audit;
- 11-trajectory materialization aggregate audit.

The planner fails closed if an input is absent; counts, costs, train-only source
boundaries, benchmark balance, strict source identity, or completed quality gates
differ. It never falls back to the narrative numbers in a handoff document.

## 1. Local implementation verification

On the Windows development host, use the available Python environment:

```powershell
cd D:\Projects\LLM\MedTrace-R1
D:\Miniconda3\python.exe -m unittest tests.test_step2_scaleup_planner -v
D:\Miniconda3\python.exe -m unittest discover
```

The formal private report cannot be produced on Windows because
`pilot_v1_real` is intentionally absent there.

## 2. Separate synchronization gate

Do not commit or push merely because offline planning was approved. After the
user separately approves a clean commit/push, synchronize the exact approved
snapshot to AutoDL with a fast-forward-only workflow. Preserve every ignored
private artifact in place.

## 3. AutoDL zero-call preflight

Do not start vLLM and do not source the network accelerator. In the shell used
for the precheck, remove model credentials so an accidental paid request cannot
be authenticated:

```bash
cd /root/autodl-tmp/MedTrace-R1
source .venv/bin/activate
unset DASHSCOPE_API_KEY OPENROUTER_API_KEY MEDTRACE_API_KEY

python -m unittest discover
python -m data_pipeline.step2_scaleup_planner --preflight-only
```

The preflight must report `status=passed`, zero network/model calls, zero GPU
inference calls, and zero training records written.

## 4. Preview and write the aggregate report

Preview without writing:

```bash
python -m data_pipeline.step2_scaleup_planner
```

After confirming the preview contains no question, ID, prompt, response, or
trajectory text, write the approved aggregate outputs:

```bash
python -m data_pipeline.step2_scaleup_planner --write-approved-report
```

Outputs:

```text
reports/step2_scaleup_budget_v1.json
reports/step2_scaleup_budget_v1.md
```

The report freezes a planning recommendation and a separately priced paid-canary
proposal. Both remain unauthorized for execution.

## 5. Required review before any paid action

Review all of the following with the user:

- observed per-stage yields and physical request counts;
- DashScope and OpenRouter cost estimates, separate stop lines, and hard caps;
- conservative GPU-hour upper bound and reconfirmed current hourly price;
- 200/300/500/1,000 negative-trajectory scenarios and PRM positive sampling quota;
- the recommended new-question and negative-candidate quotas;
- paid-canary quality gates and benchmark coverage.

Only a later, explicit approval may authorize the exact paid canary. A passing
canary still does not authorize full generation, merging the materialized
derivative into training data, or starting SFT/PRM training.
